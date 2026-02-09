from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

Base = declarative_base()

class SizeEnum(str, enum.Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class RoleEnum(str, enum.Enum):
    SALES_REP = "sales_rep"
    MANAGER = "manager"
    REGIONAL_DIRECTOR = "regional_director"

class Dealer(Base):
    __tablename__ = "dealers"
    
    dealer_id = Column(Integer, primary_key=True)
    name = Column(String)
    country = Column(String)
    city = Column(String)
    size = Column(Enum(SizeEnum))
    brands = Column(String) # Comma-separated
    avg_monthly_volume = Column(Integer)
    churn_risk_score = Column(Float)
    joined_date = Column(DateTime, default=datetime.utcnow)
    
    inventory = relationship("Inventory", back_populates="dealer")
    transactions = relationship("Transaction", back_populates="dealer")
    leads = relationship("Lead", back_populates="dealer")
    kpi_snapshots = relationship("KPISnapshot", back_populates="dealer")

class Employee(Base):
    __tablename__ = "employees"
    
    rep_id = Column(Integer, primary_key=True)
    name = Column(String)
    role = Column(Enum(RoleEnum))
    region = Column(String)
    quota = Column(Float)
    start_date = Column(DateTime)
    
    leads = relationship("Lead", back_populates="assigned_rep")

class Inventory(Base):
    __tablename__ = "inventory"
    
    car_id = Column(Integer, primary_key=True)
    dealer_id = Column(Integer, ForeignKey("dealers.dealer_id"))
    make = Column(String)
    model = Column(String)
    year = Column(Integer)
    acquisition_price = Column(Float)
    expected_resale_price = Column(Float)
    condition_score = Column(Integer) # 1-10
    location = Column(String)
    ingredients_date = Column(DateTime, default=datetime.utcnow)
    days_in_stock = Column(Integer)
    status = Column(String) # available, sold, reserved
    
    dealer = relationship("Dealer", back_populates="inventory")
    transaction = relationship("Transaction", back_populates="car", uselist=False)

class Transaction(Base):
    __tablename__ = "transactions"
    
    transaction_id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    dealer_id = Column(Integer, ForeignKey("dealers.dealer_id"))
    car_id = Column(Integer, ForeignKey("inventory.car_id"))
    sale_price = Column(Float)
    margin = Column(Float)
    channel = Column(String) # auction, direct, wholesale
    
    dealer = relationship("Dealer", back_populates="transactions")
    car = relationship("Inventory", back_populates="transaction")

class Lead(Base):
    __tablename__ = "leads"
    
    lead_id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    dealer_id = Column(Integer, ForeignKey("dealers.dealer_id"))
    source = Column(String) # website, referral, aggregation
    inquiry_text = Column(String)
    response_time_minutes = Column(Float)
    assigned_rep_id = Column(Integer, ForeignKey("employees.rep_id"))
    converted = Column(Boolean, default=False)
    conversion_date = Column(DateTime, nullable=True)
    conversion_probability = Column(Float) # AI Score
    
    dealer = relationship("Dealer", back_populates="leads")
    assigned_rep = relationship("Employee", back_populates="leads")

class KPISnapshot(Base):
    __tablename__ = "kpi_snapshots"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    dealer_id = Column(Integer, ForeignKey("dealers.dealer_id"))
    conversion_rate = Column(Float)
    avg_ticket_size = Column(Float)
    inventory_turnover = Column(Float)
    forecast_accuracy = Column(Float)
    
    dealer = relationship("Dealer", back_populates="kpi_snapshots")
