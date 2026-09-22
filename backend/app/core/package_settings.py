"""Base for the settings object a feature package owns.

Fields resolve from the package's ``env_prefix`` spelled with one trailing
underscore or with two — ``JUSTSAY_STT_MODE`` and ``JUSTSAY_STT__MODE`` alike,
in the environment and in ``.env``. The one-underscore spelling wins when both
are set.
"""

from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)


class PackageSettings(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Each declared env source followed by its doubled-prefix twin."""
        doubled_prefix = f"{settings_cls.model_config.get('env_prefix', '')}_"
        return (
            init_settings,
            env_settings,
            EnvSettingsSource(settings_cls, env_prefix=doubled_prefix),
            dotenv_settings,
            DotEnvSettingsSource(
                settings_cls,
                env_file=settings_cls.model_config.get("env_file"),
                env_file_encoding=settings_cls.model_config.get("env_file_encoding"),
                env_prefix=doubled_prefix,
            ),
            file_secret_settings,
        )
