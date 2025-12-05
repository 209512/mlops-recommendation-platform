from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin


class UserPreferCategory(Base, TimestampMixin):
    """사용자 선호 카테고리 모델"""

    __tablename__ = "user_prefer_categories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    preference_score = Column(Float, default=1.0)

    # 관계 설정
    category = relationship("Category", back_populates="user_preferences")

    def __repr__(self) -> str:
        return (
            f"<UserPreferCategory(user_id={self.user_id}, "
            f"category_id={self.category_id}, score={self.preference_score})>"
        )
