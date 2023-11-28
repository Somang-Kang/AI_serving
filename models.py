from sqlalchemy import Column, Integer, Boolean, Text, LargeBinary
from database import Base

class Categoreyes(Base):
    __tablename__ = 'categoreyes'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    category = Column(Text)
    filename = Column(Text)