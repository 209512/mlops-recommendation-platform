from datetime import datetime

from pydantic import BaseModel, Field


class CategoryBase(BaseModel):
    """카테고리 기본 스키마"""

    name: str = Field(..., min_length=1, max_length=100)


class Category(CategoryBase):
    """카테고리 응답 스키마"""

    id: int

    class Config:
        from_attributes = True


class LectureBase(BaseModel):
    """강의 기본 스키마"""

    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    instructor: str | None = None
    thumbnail_img_url: str | None = None
    platform: str = Field(..., min_length=1, max_length=50)
    external_id: str = Field(..., min_length=1, max_length=100)
    difficulty: str | None = Field(None, pattern="^(초급|중급|고급)$")
    original_price: int = Field(..., ge=0)
    discount_price: int | None = Field(None, ge=0)
    average_rating: float = Field(..., ge=0.0, le=5.0)
    duration: int | None = Field(None, ge=0)
    url_link: str | None = None


class LectureCreate(LectureBase):
    """강의 생성 스키마"""

    category_ids: list[int] = []


class LectureUpdate(BaseModel):
    """강의 업데이트 스키마"""

    title: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    instructor: str | None = None
    thumbnail_img_url: str | None = None
    difficulty: str | None = Field(None, pattern="^(초급|중급|고급)$")
    original_price: int | None = Field(None, ge=0)
    discount_price: int | None = Field(None, ge=0)
    average_rating: float | None = Field(None, ge=0.0, le=5.0)
    duration: int | None = Field(None, ge=0)
    url_link: str | None = None
    is_active: bool | None = None


class Lecture(LectureBase):
    """강의 응답 스키마"""

    id: int
    uuid: str
    categories: list[Category] = []
    is_bookmarked: bool = False
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LectureList(BaseModel):
    """강의 목록 응답 스키마"""

    lectures: list[Lecture]
    total: int
    page: int
    size: int
