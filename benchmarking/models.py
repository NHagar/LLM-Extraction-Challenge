from uuid import NAMESPACE_DNS, uuid5

from pydantic import BaseModel


class Newsletter(BaseModel):
    committee: str
    name: str
    email: str
    subject: str
    date: str
    year: int
    month: int
    day: int
    hour: int
    minute: int
    domain: str
    body: str
    party: str
    disclaimer: bool

    @property
    def uuid(self) -> str:
        unique_str = f"{self.committee}-{self.name}-{self.email}-{self.subject}-{self.date}-{self.year}-{self.month}-{self.day}-{self.hour}-{self.minute}-{self.domain}"
        return str(uuid5(NAMESPACE_DNS, unique_str))


class ExtractedCommittee(BaseModel):
    newsletter_uuid: str
    committee: str | None


class EvalData(BaseModel):
    extracted_committees: list[ExtractedCommittee]
