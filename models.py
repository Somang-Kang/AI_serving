from sqlalchemy import Column, Integer, Boolean, Text, LargeBinary
from database import Base

class Categoreyes(Base):
    __tablename__ = 'categoreyes'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    category = Column(Text)
    filename = Column(Text)

class Human(Base):
    __tablename__ = 'human'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    filename = Column(Text)

class Animal(Base):
    __tablename__ = 'animal'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    filename = Column(Text)

class Nature(Base):
    __tablename__ = 'nature'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    filename = Column(Text)

class Food(Base):
    __tablename__ = 'food'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    filename = Column(Text)

class Others(Base):
    __tablename__ = 'others'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    filename = Column(Text)

class Docs(Base):
    __tablename__ = 'docs'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    filename = Column(Text)

