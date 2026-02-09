import random
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
import math
import logging

# Add parent directory to path to import schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings
from database.schema import Base, Dealer, Employee, Inventory, Transaction, Lead, SizeEnum, RoleEnum, KPISnapshot

settings = get_settings()
engine = create_engine(settings.DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

fake = Faker('de_DE')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_seasonality_factor(date):
    """
    Returns a seasonality factor based on month.
    Peak in March (3) and September (9).
    Low in August (8) and December (12).
    """
    month = date.month
    # Simple sine wave approximation + noise
    base = 1.0
    seasonality = 0.2 * math.sin((month - 1) * math.pi / 6)  # Period of 12 months
    # Adjust for specific peaks if needed, but sine wave gives smooth transition
    # Month 3 (March) -> sin(2pi/6) ~ 0.86 (High)
    # Month 9 (Sept) -> sin(8pi/6) ~ -0.86 (Wait, simple sine is simple). 
    
    # Custom multipliers for automotive:
    if month in [3, 4, 5, 9, 10]:
        return random.uniform(1.1, 1.3)
    elif month in [8, 12]:
        return random.uniform(0.7, 0.9)
    else:
        return random.uniform(0.9, 1.1)

def generate_dealers(n=20):
    logger.info(f"Generating {n} dealers...")
    brands_list = ["Volkswagen", "BMW", "Mercedes-Benz", "Audi", "Ford", "Opel", "Skoda"]
    dealers = []
    for _ in range(n):
        size = random.choice(list(SizeEnum))
        churn_risk = random.uniform(0, 1)
        
        dealer = Dealer(
            name=fake.company(),
            country="Germany",
            city=fake.city(),
            size=size,
            brands=",".join(random.sample(brands_list, k=random.randint(1, 3))),
            avg_monthly_volume=random.randint(20, 150) if size == SizeEnum.LARGE else random.randint(5, 40),
            churn_risk_score=churn_risk,
            joined_date=fake.date_time_between(start_date='-5y', end_date='-2y')
        )
        dealers.append(dealer)
    session.add_all(dealers)
    session.commit()
    return dealers

def generate_employees(dealers):
    logger.info("Generating employees...")
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
    logger.info(f"Generating inventory and transactions for {years} years...")
    
    start_date = datetime.now() - timedelta(days=365*years)
    end_date = datetime.now()
    
    cars = []
    transactions = []
    leads = []
    
    for dealer in dealers:
        # Determine dealer volume trend
        # If High churn risk, volume decreases over time
        volume_trend = -0.5 if dealer.churn_risk_score > 0.7 else 0.1
        
        # Base annual volume
        annual_volume = dealer.avg_monthly_volume * 12
        
        # Iterate through months
        current_date = start_date
        while current_date < end_date:
            month_factor = get_seasonality_factor(current_date)
            
            # Apply churn volume degradation
            months_passed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
            churn_factor = max(0.2, 1 + (volume_trend * months_passed / 36)) # Decay over 3 years
            
            monthly_target = int(dealer.avg_monthly_volume * month_factor * churn_factor)
            
            for _ in range(monthly_target):
                # Inventory Creation
                acquisition_date = fake.date_time_between(start_date=current_date, end_date=current_date + timedelta(days=28))
                if acquisition_date > end_date: continue
                
                make = random.choice(dealer.brands.split(","))
                
                car = Inventory(
                    dealer_id=dealer.dealer_id,
                    make=make,
                    model=fake.word().capitalize(),
                    year=random.randint(2018, 2024),
                    acquisition_price=random.randint(10000, 50000),
                    condition_score=random.randint(1, 10),
                    location=dealer.city,
                    ingredients_date=acquisition_date,
                    days_in_stock=random.randint(1, 180),
                    status="available"
                )
                
                # Transaction Logic
                # Older cars (>90 days) get price discounted
                selling_probability = 0.8
                if dealer.churn_risk_score > 0.8:
                    selling_probability = 0.4 # Churning dealers sell less
                
                if random.random() < selling_probability:
                    car.status = "sold"
                    sale_date = car.ingredients_date + timedelta(days=random.randint(1, car.days_in_stock))
                    
                    if sale_date <= end_date:
                        # Price Logic
                        days_held = (sale_date - car.ingredients_date).days
                        aging_discount = 0.95 if days_held > 90 else 1.0
                        
                        margin_pct = random.uniform(0.05, 0.15) * aging_discount
                        sale_price = car.acquisition_price * (1 + margin_pct)
                        
                        transaction = Transaction(
                            date=sale_date,
                            dealer_id=dealer.dealer_id,
                            car=car,
                            sale_price=sale_price,
                            margin=sale_price - car.acquisition_price,
                            channel=random.choice(["auction", "direct", "wholesale"])
                        )
                        transactions.append(transaction)
                    else:
                         car.status = "available"
                
                cars.append(car)
            
            # Next month
            current_date += timedelta(days=30)
            
        # Leads Generation (correlated with volume)
        for _ in range(int(annual_volume * years * 0.5)): # 50% conversion rate approx
             lead_date = fake.date_time_between(start_date=start_date, end_date=end_date)
             lead = Lead(
                created_at=lead_date,
                dealer_id=dealer.dealer_id,
                source=random.choice(["website", "referral", "email"]),
                inquiry_text=fake.sentence(),
                response_time_minutes=random.randint(5, 120),
                assigned_rep_id=random.choice(employees).rep_id,
                conversion_probability=random.uniform(0.1, 0.9)
            )
             # Logic for conversion
             if lead.conversion_probability > 0.6:
                 lead.converted = True
                 lead.conversion_date = lead_date + timedelta(days=random.randint(1, 20))
             else:
                 lead.converted = False
                 
             leads.append(lead)

    # Bulk insert (in chunks if needed, but 3 years manageable)
    logger.info(f"Saving {len(cars)} cars, {len(transactions)} transactions, {len(leads)} leads...")
    session.add_all(cars)
    session.add_all(transactions)
    session.add_all(leads)
    session.commit()
    logger.info("Data generation complete.")

def init_db():
    Base.metadata.drop_all(engine) # Reset DB for clean generation
    Base.metadata.create_all(engine)
    logger.info("Database tables recreated.")

if __name__ == "__main__":
    init_db()
    dealers = generate_dealers()
    employees = generate_employees(dealers)
    generate_inventory_and_transactions(dealers, employees)
