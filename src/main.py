import asyncio
from viam.module.module import Module

import sdk_log_channel
from models.color_correction import ColorCorrection as ColorCorrectionModel
from models.ptp import PTP as PTPModel


if __name__ == '__main__':
    # Before the module connects to viam-server: module logs get their own
    # connection instead of sharing (and corrupting) the module's main one.
    # CONSULT-2296 / CONSULT-2298; see sdk_log_channel.py.
    sdk_log_channel.install()
    asyncio.run(Module.run_from_registry())
