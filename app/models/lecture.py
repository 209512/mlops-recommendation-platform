from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin

if TYPE_CHECKING:
    pass


class Category(Base, TimestampMixin):
    """강의 카테고리 모델"""

    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    lecture_categories = relationship(
        "LectureCategory", back_populates="category", cascade="all, delete-orphan"
    )
    lectures = relationship("Lecture", secondary="lecture_categories", viewonly=True)
    user_preferences = relationship("UserPreferCategory", back_populates="category")

    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}')>"


class Lecture(Base, TimestampMixin):
    """강의 모델"""

    __tablename__ = "lectures"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid4()), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    instructor = Column(String(100), nullable=True)
    thumbnail_img_url = Column(String(500), nullable=True)
    platform = Column(String(50), nullable=False)
    difficulty = Column(String(20), nullable=True)
    original_price = Column(Integer, nullable=True)
    discount_price = Column(Integer, nullable=True)
    rating = Column(Float, nullable=True)
    review_count = Column(Integer, nullable=True)
    url_link = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)

    lecture_categories = relationship(
        "LectureCategory", back_populates="lecture", cascade="all, delete-orphan"
    )
    categories = relationship("Category", secondary="lecture_categories", viewonly=True)
    bookmarks = relationship("Bookmark", back_populates="lecture", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Lecture(id={self.id}, title='{self.title[:50]}...')>"

    @property
    def is_discounted(self) -> bool:
        """할인 여부 확인"""
        return bool(self.discount_price is not None and self.discount_price < self.original_price)

    @property
    def discount_rate(self) -> float:
        """할인율 계산"""
        if not self.is_discounted:
            return 0.0
        return (
            round((1 - self.discount_price / self.original_price) * 100, 1)
            if self.original_price > 0
            else 0.0
        )


class LectureCategory(Base, TimestampMixin):
    """강의-카테고리 연결 모델"""

    __tablename__ = "lecture_categories"

    lecture_id = Column(Integer, ForeignKey("lectures.id"), primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), primary_key=True)

    # 추가 속성 (필요시 확장 가능)
    is_primary = Column(Boolean, default=False)
    weight = Column(Float, default=1.0)

    lecture = relationship("Lecture", back_populates="lecture_categories")
    category = relationship("Category", back_populates="lecture_categories")

    def __repr__(self) -> str:
        return f"<LectureCategory(lecture_id={self.lecture_id}, category_id={self.category_id})>"
