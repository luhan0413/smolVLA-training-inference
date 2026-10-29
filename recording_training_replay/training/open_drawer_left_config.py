from gr00t.experiment.data_config import OxeDroidDataConfig


class OpenDrawerLeftDataConfig(OxeDroidDataConfig):
    """OXE settings for the UR7e dataset's single arm camera."""

    video_keys = ["video.arm_image"]
