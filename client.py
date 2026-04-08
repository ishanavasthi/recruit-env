"""RecruitEnv client for OpenEnv compatibility."""

from openenv_core import GenericEnvClient


class RecruitEnvClient(GenericEnvClient):
    """Thin wrapper around GenericEnvClient with RecruitEnv defaults."""

    def __init__(self, base_url: str = "https://heyavasthi-recruitenv.hf.space"):
        super().__init__(base_url=base_url)
