"""Phase handler package.

Each module in this package implements one phase of the IVR call flow.
``call_phases`` contains the orchestrator that drives the call through
those phases as well as the shared ``_speak`` / ``_listen`` helpers used
by every phase handler.
"""
