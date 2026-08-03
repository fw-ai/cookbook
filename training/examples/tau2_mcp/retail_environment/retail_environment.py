#!/usr/bin/env python3
"""
Retail Environment for τ²-Bench Integration

This module implements a RetailEnvironment that integrates the τ²-Bench simulation
pattern (Agent/User/Environment communication) with the MCP-Gym framework.
"""

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vendor.tau2.domains.retail.data_model import RetailDB
from vendor.tau2.domains.retail.tools import RetailTools

logger = logging.getLogger(__name__)

RETAIL_DB_PATH = Path(__file__).parent / "db.json"


class RetailEnvironment:
    """
    Retail environment that integrates τ²-Bench simulation pattern
    with MCP-Gym framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db = None
        self.airline_tools = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state"""
        self.db = RetailDB.load(RETAIL_DB_PATH)
        self.retail_tools = RetailTools(self.db)

        return {}, {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Perform one step of the τ²-Bench simulation.
        """

        action_name = action.get("action", "")
        parameters = action.get("parameters", {})

        result = self._execute_retail_action(action_name, parameters)

        # In tau2-bench, if there's a simulated user, the agent cannot terminate the rollout, and there are no per step rewards.
        observation = result
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _execute_retail_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action using retail tools."""
        action_map = {
            "calculate": self.retail_tools.calculate,
            "cancel_pending_order": self.retail_tools.cancel_pending_order,
            "exchange_delivered_order_items": self.retail_tools.exchange_delivered_order_items,
            "find_user_id_by_name_zip": self.retail_tools.find_user_id_by_name_zip,
            "find_user_id_by_email": self.retail_tools.find_user_id_by_email,
            "get_order_details": self.retail_tools.get_order_details,
            "get_product_details": self.retail_tools.get_product_details,
            "get_user_details": self.retail_tools.get_user_details,
            "list_all_product_types": self.retail_tools.list_all_product_types,
            "modify_pending_order_address": self.retail_tools.modify_pending_order_address,
            "modify_pending_order_items": self.retail_tools.modify_pending_order_items,
            "modify_pending_order_payment": self.retail_tools.modify_pending_order_payment,
            "modify_user_address": self.retail_tools.modify_user_address,
            "return_delivered_order_items": self.retail_tools.return_delivered_order_items,
            "transfer_to_human_agents": self.retail_tools.transfer_to_human_agents,
        }

        if action_name in action_map:
            tool_method = action_map[action_name]
            # Call the tool method with parameters
            if parameters:
                return tool_method(**parameters)
            else:
                return tool_method()
        else:
            return {"error": f"Unknown action: {action_name}"}

    @property
    def observation_space(self):
        """Return the observation space"""
        return {}

    @property
    def action_space(self):
        """Return the action space"""
        return {}

    def render(self, mode="human"):
        """Render the environment"""
        pass

    def close(self):
        """Close the environment"""
        pass
