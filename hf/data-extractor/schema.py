from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str
    role: str | None = Field(default=None)
    context: str | None = Field(default=None)


class Place(BaseModel):
    name: str
    context: str | None = Field(default=None)


class Date(BaseModel):
    date: str
    event: str | None = Field(default=None)


class ExtractedDocument(BaseModel):
    title: str | None = Field(default=None)
    topic: str
    people: list[Person] = Field(default_factory=list)
    places: list[Place] = Field(default_factory=list)
    dates: list[Date] = Field(default_factory=list)
    summary: str
