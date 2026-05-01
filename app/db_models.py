import uuid

from sqlalchemy import (
    Boolean,
    TIMESTAMP,
    Column,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.routes.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    firebase_uid = Column(Text, unique=True, nullable=False)
    email = Column(Text)
    name = Column(Text)
    photo_object_name = Column(Text)
    phone_number = Column(Text)
    years_of_experience = Column(SmallInteger)
    acres = Column(Float)
    primary_crops = Column(JSONB)
    soil_type = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    uploads = relationship("Upload", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")
    ai_reports = relationship("AiReport", back_populates="user")
    crop_plans = relationship("CropPlan", back_populates="user")


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    object_name = Column(Text, nullable=False)
    bucket = Column(Text, nullable=False)
    original_filename = Column(Text)
    content_type = Column(Text)
    file_size_bytes = Column(Integer)
    status = Column(Text, server_default="uploaded")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="uploads")
    prediction = relationship("Prediction", back_populates="upload", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    upload_id = Column(
        UUID(as_uuid=True), ForeignKey("uploads.id"), nullable=False, index=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    model_version = Column(Text, server_default="crop136_b3_final")
    inference_ms = Column(Float)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="predictions")
    upload = relationship("Upload", back_populates="prediction")
    prediction_results = relationship("PredictionResult", back_populates="prediction")
    ai_report = relationship("AiReport", back_populates="prediction", uselist=False)


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    prediction_id = Column(
        UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False
    )
    rank = Column(SmallInteger, nullable=False)
    label = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="prediction_results")


class AiReport(Base):
    __tablename__ = "ai_reports"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    prediction_id = Column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    report_text = Column(Text, nullable=False)

    is_diseased = Column(Boolean, nullable=False, server_default=text("false"))
    disease_name = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    severity = Column(Text, nullable=True)
    treatments = Column(JSONB, nullable=True)
    product_links = Column(JSONB, nullable=True)
    expert_analysis = Column(JSONB, nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="ai_reports")
    prediction = relationship("Prediction", back_populates="ai_report")


class CropPlan(Base):
    __tablename__ = "crop_plans"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    location = Column(Text, nullable=True)
    weather_summary = Column(Text, nullable=True)
    soil_type = Column(Text, nullable=True)
    recommended_crops = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="crop_plans")
