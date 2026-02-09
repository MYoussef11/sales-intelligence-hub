import random
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import Base, Dealer, Employee, Inventory, Transaction, Lead, SizeEnum, RoleEnum, KPISnapshot

# Database Connection
DATABASE_URL = "postgresql://admin:admin123@localhost:5432/sales_intelligence"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

fake = Faker('de_DE') # German locale for realism

def generate_dealers(n=20):
    print(f"Generating {n} dealers...")
    brands_list = ["Volkswagen", "BMW", "Mercedes-Benz", "Audi", "Ford", "Opel", "Skoda"]
    dealers = []
    for _ in range(n):
        size = random.choice(list(SizeEnum))
        dealer = Dealer(
            name=fake.company(),
            country="Germany",
            city=fake.city(),
            size=size,
            brands=",".join(random.sample(brands_list, k=random.randint(1, 3))),
            avg_monthly_volume=random.randint(20, 150) if size == SizeEnum.LARGE else random.randint(5, 40),
            churn_risk_score=random.uniform(0, 1),
            joined_date=fake.date_time_between(start_date='-5y', end_date='-2y')
        )
        dealers.append(dealer)
    session.add_all(dealers)
    session.commit()
    return dealers

def generate_employees(dealers):
    print("Generating employees...")
    employees = []
    # Internal Sales Reps
    for _ in range(5):
        emp = Employee(
            name=fake.name(),
            role=RoleEnum.SALES_REP,
            region=random.choice(["North", "South", "East", "West"]),
            quota=random.randint(500000, 1000000),
            start_date=fake.date_time_between(start_date='-4y', end_date='-1y')
        )
        employees.append(emp)
    
    session.add_all(employees)
    session.commit()
    return employees

def generate_inventory_and_transactions(dealers, employees, years=3):
    print(f"Generating inventory and transactions for {years} years...")
    
    start_date = datetime.now() - timedelta(days=365*years)
    end_date = datetime.now()
    
    cars = []
    transactions = []
    leads = []
    
    # Simulation Loop (simplified per dealer)
    for dealer in dealers:
        # Generate Inventory
        for _ in range(random.randint(50, 200)): # Inventory per dealer
            acquisition_date = fake.date_time_between(start_date=start_date, end_date=end_date)
            make = random.choice(dealer.brands.split(","))
            model = fake.word().capitalize() # Simplified
            
            car = Inventory(
                dealer_id=dealer.dealer_id,
                make=make,
                model=model,
                year=random.randint(2018, 2024),
                acquisition_price=random.randint(10000, 50000),
                condition_score=random.randint(1, 10),
                location=dealer.city,
                ingredients_date=acquisition_date,
                days_in_stock=random.randint(1, 180),
                status="available" # Default
            )
            
            # Simulate Sale (Transaction)
            if random.random() > 0.3: # 70% sell rate
                car.status = "sold"
                car.expected_resale_price = car.acquisition_price * 1.15
                
                sale_date = car.ingredients_date + timedelta(days=car.days_in_stock)
                if sale_date > end_date:
                    car.status = "available" # Didn't sell yet
                else:
                    transaction = Transaction(
                        date=sale_date,
                        dealer_id=dealer.dealer_id,
                        car=car,
                        sale_price=car.expected_resale_price * random.uniform(0.95, 1.05),
                        margin=0, # Calculated later
                        channel=random.choice(["auction", "direct", "wholesale"])
                    )
                    transaction.margin = transaction.sale_price - car.acquisition_price
                    transactions.append(transaction)
            
            cars.append(car)

        # Generate Leads
        for _ in range(random.randint(100, 500)):
            lead_date = fake.date_time_between(start_date=start_date, end_date=end_date)
            lead = Lead(
                created_at=lead_date,
                dealer_id=dealer.dealer_id,
                source=random.choice(["website", "referral", "email"]),
                inquiry_text=fake.sentence(),
                response_time_minutes=random.randint(5, 120),
                assigned_rep_id=random.choice(employees).rep_id,
                converted=random.choice([True, False]),
                conversion_probability=random.uniform(0.1, 0.9)
            )
            if lead.converted:
                lead.conversion_date = lead_date + timedelta(days=random.randint(1, 30))
            leads.append(lead)

    session.add_all(cars)
    session.add_all(transactions)
    session.add_all(leads)
    session.commit()
    print("Data generation complete.")

def init_db():
    Base.metadata.create_all(engine)
    print("Database tables created.")

if __name__ == "__main__":
    init_db()
    dealers = generate_dealers()
    employees = generate_employees(dealers)
    generate_inventory_and_transactions(dealers, employees)
