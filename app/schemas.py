from typing import Optional, Union
from uuid import UUID
from datetime import datetime
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictRequest(BaseModel):
    uid: str
    object_name: str


class BatchPredictRequest(BaseModel):
    object_names: list[str]


class PredictResponse(BaseModel):
    object_name: str
    top_predictions: list[Prediction]
    inference_ms: float
    model_version: str = "crop136_b3_final"
    prediction_id: Optional[UUID] = None


class UploadResponse(BaseModel):
    status: str
    object_name: str
    bucket: str
    message: str


class UserUploadsResponse(BaseModel):
    uid: str
    images: list[str]


class HealthResponse(BaseModel):
    status: str
    num_classes: int
    model_load_time_s: Optional[float]
    gpu_available: bool


class RegisterUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    firebase_uid: str = Field(
        validation_alias=AliasChoices("firebase_uid", "firebaseUid", "uid")
    )
    email: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("phone_number", "phoneNumber"),
    )
    years_of_experience: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "years_of_experience",
            "yearsOfExperience",
            "yearsOfExp",
        ),
    )
    acres: Optional[float] = None
    primary_crops: Optional[Union[list[str], str]] = Field(
        default=None,
        validation_alias=AliasChoices("primary_crops", "primaryCrops"),
    )
    soil_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("soil_type", "soilType"),
    )


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    firebase_uid: str
    email: Optional[str]
    name: Optional[str] = None
    photo_object_name: Optional[str] = None
    phone_number: Optional[str] = None
    years_of_experience: Optional[int] = None
    acres: Optional[float] = None
    primary_crops: Optional[list[str]] = None
    soil_type: Optional[str] = None


class UserProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    firebase_uid: str
    email: Optional[str]
    name: Optional[str] = None
    photo_object_name: Optional[str] = None
    photo_url: Optional[str] = None
    phone_number: Optional[str] = None
    years_of_experience: Optional[int] = None
    acres: Optional[float] = None
    primary_crops: Optional[list[str]] = None
    soil_type: Optional[str] = None


# ─── Activity Schemas ───────────────────────────────────────────────

class ActivityPredictionResult(BaseModel):
    rank: int
    label: str
    confidence: float

class ActivityResponse(BaseModel):
    id: UUID
    image_name: str
    image_url: str | None = None
    inference_ms: float
    created_at: datetime | None
    results: list[ActivityPredictionResult]


# ─── AI Agent Schemas ────────────────────────────────────────────────

class LocationItem(BaseModel):
    lat: float
    lon: float

class AiReportRequest(BaseModel):
    prediction_id: UUID
    location: Optional[LocationItem] = Field(None, description="Current location coordinates")
    crop: Optional[str] = Field(None, description="Crop name for expert analysis")


class TreatmentStep(BaseModel):
    step: str = Field(..., description="Treatment step description")
    dosage: Optional[str] = Field(None, description="Dosage information if applicable")


class ReportAgentOutput(BaseModel):
    is_diseased: bool = Field(..., description="Whether the crop is diseased")
    disease_name: Optional[str] = Field(None, description="Name of the disease if diseased")
    severity: Optional[str] = Field(None, description="Severity level: low, medium, or high")
    report_text: str = Field(..., description="Farmer-friendly diagnostic report")
    treatments: Optional[list[TreatmentStep]] = Field(None, description="Step-by-step treatment plan")


class WebAgentOutput(BaseModel):
    product_links: list[str] = Field(default_factory=list, description="Product purchase URLs")


class ExpertAgentOutput(BaseModel):
    cause: str = Field(..., description="Why this disease occurred")
    weather_impact: str = Field(..., description="How weather affects the disease")
    prevention: str = Field(..., description="Preventive measures")


class AiReportResponse(BaseModel):
    prediction_id: UUID
    report: ReportAgentOutput
    expert_analysis: Optional[ExpertAgentOutput] = None
    product_links: list[str] = Field(default_factory=list)


class CropPlanRequest(BaseModel):
    uid: str
    location: Union[LocationItem, str]


class CropSuggestion(BaseModel):
    crop: str = Field(..., description="Recommended crop name")
    reason: str = Field(..., description="Why this crop is recommended")


class CropPlanResponse(BaseModel):
    deduced_soil_type: str = Field(..., description="The type of soil automatically detected based on the region")
    recommended_crops: list[CropSuggestion] = Field(..., description="List of crop suggestions")
