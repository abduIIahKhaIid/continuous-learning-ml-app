from pydantic import BaseModel, ConfigDict, Field


class DataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_1: float = Field(allow_inf_nan=False)
    feature_2: float = Field(allow_inf_nan=False)
    feature_3: float = Field(allow_inf_nan=False)
    label: int | None = None


class DataResponse(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    label: int | None
