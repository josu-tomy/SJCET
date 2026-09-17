"""
MachineGuard — Sensor Interface Abstraction
Defines the architectural contract and adapter interfaces for ingesting hardware sensor telemetry.

Planned Architecture:
  Physical Sensor / IoT Edge (ESP32, Current Transducer, Accelerometer, Thermocouple)
         ↓  [MQTT / Serial / CoAP / REST]
  sensor_interface.py (SensorGateway / TelemetryAdapter)
         ↓
  MachineGuard Core (appliance-specific parsing & normalization)
         ↓
  ML Anomaly & Risk Detection Engine
         ↓
  Real-Time Dashboard & Alert System

NOTE: Physical hardware connection is planned for subsequent deployment phases.
      Currently operating in prototype manual telemetry ingestion mode.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import datetime


@dataclass
class TelemetryPacket:
    """Standardized telemetry payload contract for physical IoT sensor ingestion."""
    device_id: str
    appliance_type: str
    timestamp: datetime.datetime
    measurements: Dict[str, float]
    metadata: Dict[str, Any]
    is_simulated: bool = False


class SensorGateway:
    """
    Abstract sensor gateway interface for future IoT and edge device connectivity.
    Standardizes ingestion from various physical protocols (MQTT, HTTP, Serial).
    """

    def __init__(self):
        self._connected_devices = {
            "ac_unit_01": {"type": "Air Conditioner", "status": "Ready for hardware pairing"},
            "fridge_01": {"type": "Refrigerator", "status": "Ready for hardware pairing"},
            "washer_01": {"type": "Washing Machine", "status": "Ready for hardware pairing"},
            "tv_01": {"type": "Television", "status": "Ready for hardware pairing"},
            "pump_01": {"type": "Water Pump", "status": "Ready for hardware pairing"},
            "fan_01": {"type": "Ceiling Fan", "status": "Ready for hardware pairing"},
        }

    def list_configured_devices(self) -> Dict[str, Dict[str, str]]:
        """Returns list of pre-configured IoT device endpoints."""
        return self._connected_devices

    def get_sensor_readings(self, device_id: str) -> Optional[TelemetryPacket]:
        """
        Ingests real-time sensor measurements from a physical device endpoint.
        Returns a structured TelemetryPacket or None if hardware is uncoupled.
        """
        if device_id not in self._connected_devices:
            raise KeyError(f"Device identifier '{device_id}' is not configured in gateway.")

        # Documented architectural placeholder: hardware connection pending
        device_meta = self._connected_devices[device_id]
        return TelemetryPacket(
            device_id=device_id,
            appliance_type=device_meta["type"],
            timestamp=datetime.datetime.now(),
            measurements={},
            metadata={"connection": "Awaiting physical sensor integration", "protocol": "Unbound"},
            is_simulated=False,
        )


# Singleton Gateway Instance
sensor_gateway = SensorGateway()


def get_sensor_readings(device_id: str) -> Optional[TelemetryPacket]:
    """Top-level convenience accessor for sensor gateway."""
    return sensor_gateway.get_sensor_readings(device_id)

