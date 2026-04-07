from openenv_core.client import HTTPEnvClient
from env.models import Action, Observation, Reward, EpisodeState


class RecruitEnvClient(HTTPEnvClient):
    def __init__(self, base_url: str = "http://localhost:7860"):
        super().__init__(base_url=base_url)

    async def reset(self, task_id: str = "easy", seed: int = 42):
        return await self._post("/reset", {"task_id": task_id, "seed": seed})

    async def step(self, action: dict):
        return await self._post("/step", action)

    async def state(self):
        return await self._get("/state")
