from sqlalchemy import Column, Float, Integer, String, Text

from app.models.base import Base, TimestampMixin


class LectureSearchLog(Base, TimestampMixin):
    """강의 검색 로그 모델 - 검색 행동을 암시적 피드백으로 활용"""

    __tablename__ = "lecture_search_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    search_query = Column(String(255), nullable=False)
    clicked_lecture_ids = Column(Text, nullable=True)  # JSON 형식으로 저장
    search_duration = Column(Float, nullable=True)  # 검색 소요 시간(초)
    result_count = Column(Integer, default=0)

    def __repr__(self) -> str:
        return (
            f"<LectureSearchLog(id={self.id}, user_id={self.user_id}, query='{self.search_query}')>"
        )
