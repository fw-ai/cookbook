#!/usr/bin/env python3
"""
Airline Environment for τ²-Bench Integration

This module implements an AirlineEnvironment that integrates the τ²-Bench simulation
pattern (Agent/User/Environment communication) with the MCP-Gym framework.
"""

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vendor.tau2.domains.airline.data_model import FlightDB
from vendor.tau2.domains.airline.tools import AirlineTools

logger = logging.getLogger(__name__)

AIRLINE_DB_PATH = Path(__file__).parent / "db.json"


class AirlineEnvironment:
    """
    Airline environment that integrates τ²-Bench simulation pattern
    with MCP-Gym framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db = None
        self.airline_tools = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state"""
        logger.info("🔄 Resetting airline environment - reloading database from disk")
        self.db = FlightDB.load(AIRLINE_DB_PATH)
        self.airline_tools = AirlineTools(self.db)

        return {}, {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Perform one step of the τ²-Bench simulation.
        """

        action_name = action.get("action", "")
        parameters = action.get("parameters", {})

        result = self._execute_airline_action(action_name, parameters)

        # In tau2-bench, if there's a simulated user, the agent cannot terminate the rollout, and there are no per step rewards.

        return result, 0.0, False, False, {}

    def _calculate_reward(self):
        """Calculate the reward for the entire conversation."""
        pass

    def close(self):
        """Clean up environment resources"""
        pass

    def _execute_airline_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action using airline tools."""
        action_map = {
            "book_reservation": self.airline_tools.book_reservation,
            "cancel_reservation": self.airline_tools.cancel_reservation,
            "get_reservation_details": self.airline_tools.get_reservation_details,
            "get_user_details": self.airline_tools.get_user_details,
            "list_all_airports": self.airline_tools.list_all_airports,
            "search_direct_flight": self.airline_tools.search_direct_flight,
            "search_onestop_flight": self.airline_tools.search_onestop_flight,
            "send_certificate": self.airline_tools.send_certificate,
            "transfer_to_human_agents": self.airline_tools.transfer_to_human_agents,
            "calculate": self.airline_tools.calculate,
            "get_flight_status": self.airline_tools.get_flight_status,
            "update_reservation_baggages": self.airline_tools.update_reservation_baggages,
            "update_reservation_flights": self.airline_tools.update_reservation_flights,
            "update_reservation_passengers": self.airline_tools.update_reservation_passengers,
        }

        if action_name not in action_map:
            return {"ok": False, "error": f"Unknown action: {action_name}", "action": action_name}

        tool_method = action_map[action_name]
        try:
            # Call the tool method with parameters
            result = tool_method(**parameters) if parameters else tool_method()
            return {"ok": True, "action": action_name, "result": result}
        except Exception as exc:
            # Keep tool errors JSON-serializable so MCP clients can parse reliably.
            return {
                "ok": False,
                "action": action_name,
                "error": str(exc),
                "parameters": parameters,
            }
