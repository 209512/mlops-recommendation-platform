from sqlalchemy import Column, DateTime, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin


class Bookmark(Base, TimestampMixin):
    """강의 북마크 모델 - 기존 LectureBookmark에서 변환"""

    __tablename__ = "bookmarks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id"), nullable=False, index=True)
    bookmarked_at = Column(DateTime, default=func.now(), nullable=False)

    # 관계 설정
    user = relationship("User", back_populates="bookmarks")
    lecture = relationship("Lecture", back_populates="bookmarks")

    # 복합 유니크 제약조건 - 동일 사용자가 동일 강의를 중복 북마크 방지
    __table_args__ = (
        UniqueConstraint("user_id", "lecture_id", name="unique_user_lecture_bookmark"),
    )

    def __repr__(self) -> str:
        return f"<Bookmark(id={self.id}, user_id={self.user_id}, lecture_id={self.lecture_id})>"
