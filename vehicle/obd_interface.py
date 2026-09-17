"""
Vehicle Maintenance AI — OBD-II Hardware Interface & Telemetry Abstraction.

Provides clean hardware abstraction for automotive On-Board Diagnostics:
- Physical ELM327 adapter communication (Bluetooth RFCOMM, USB Serial, WiFi TCP)
- Standard SAE J1979 Mode 01 PID queries and AT command sequences
- Simulated adapter for offline replay, edge-testing, and automated demonstrations
- Real-time streaming generator for continuous telemetry feeds
"""

import os
import sys
import time
import enum
import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Generator, Union

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class OBDConnectionStatus(enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    SIMULATED = "SIMULATED"
    ERROR = "ERROR"


class OBDProtocol(enum.Enum):
    AUTO = "0 - Automatic"
    SAE_J1850_PWM = "1 - SAE J1850 PWM (41.6 kbaud)"
    SAE_J1850_VPW = "2 - SAE J1850 VPW (10.4 kbaud)"
    ISO_9141_2 = "3 - ISO 9141-2 (5 baud init, 10.4 kbaud)"
    ISO_14230_4_KWP_5BAUD = "4 - ISO 14230-4 KWP (5 baud init)"
    ISO_14230_4_KWP_FAST = "5 - ISO 14230-4 KWP (fast init)"
    ISO_15765_4_CAN_11BIT_500K = "6 - ISO 15765-4 CAN (11 bit ID, 500 kbaud)"
    ISO_15765_4_CAN_29BIT_500K = "7 - ISO 15765-4 CAN (29 bit ID, 500 kbaud)"
    ISO_15765_4_CAN_11BIT_250K = "8 - ISO 15765-4 CAN (11 bit ID, 250 kbaud)"


@dataclass
class OBDTelemetryFrame:
    """Represents a single unified snapshot of vehicle powertrain sensors."""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    engine_rpm: float = 0.0
    speed_kmh: float = 0.0
    coolant_temp_c: float = 85.0
    engine_load_pct: float = 30.0
    throttle_pos_pct: float = 15.0
    intake_air_temp_c: float = 30.0
    manifold_pressure_kpa: Optional[float] = None
    battery_voltage: Optional[float] = 13.8
    trouble_codes: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts frame to standard dictionary matching vehicle AI model input format."""
        return {
            "TIMESTAMP": self.timestamp,
            "ENGINE_RPM": float(self.engine_rpm),
            "SPEED": float(self.speed_kmh),
            "ENGINE_COOLANT_TEMP": float(self.coolant_temp_c),
            "ENGINE_LOAD": float(self.engine_load_pct),
            "THROTTLE_POS": float(self.throttle_pos_pct),
            "AIR_INTAKE_TEMP": float(self.intake_air_temp_c),
            "TROUBLE_CODES": self.trouble_codes[0] if self.trouble_codes else None,
            "BATTERY_VOLTAGE": self.battery_voltage,
            "IS_SIMULATED": self.is_simulated
        }


class BaseOBDAdapter:
    """Abstract interface defining required hardware communication contracts."""

    def connect(self, port: Optional[str] = None, baudrate: int = 38400, timeout: float = 2.0) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def read_frame(self) -> OBDTelemetryFrame:
        raise NotImplementedError

    def get_trouble_codes(self) -> List[str]:
        raise NotImplementedError

    def clear_trouble_codes(self) -> bool:
        raise NotImplementedError


class SimulatedOBDAdapter(BaseOBDAdapter):
    """
    Simulation adapter that can replay recorded driving trips or generate
    realistic synthetic driving cycles with fault injection support.
    """

    def __init__(self, replay_csv_path: Optional[str] = None):
        self.replay_csv_path = replay_csv_path
        self._df = None
        self._current_index = 0
        self._status = OBDConnectionStatus.DISCONNECTED
        self._fault_injection: Optional[str] = None

        if self.replay_csv_path and os.path.exists(self.replay_csv_path):
            import pandas as pd
            self._df = pd.read_csv(self.replay_csv_path, low_memory=False)

    def connect(self, port: Optional[str] = "SIMULATOR_BUS", baudrate: int = 38400, timeout: float = 1.0) -> bool:
        self._status = OBDConnectionStatus.SIMULATED
        self._current_index = 0
        return True

    def disconnect(self) -> None:
        self._status = OBDConnectionStatus.DISCONNECTED

    def is_connected(self) -> bool:
        return self._status in [OBDConnectionStatus.CONNECTED, OBDConnectionStatus.SIMULATED]

    def inject_fault(self, dtc_code: Optional[str]) -> None:
        """Injects synthetic DTC or fault state into simulation stream."""
        self._fault_injection = dtc_code

    def read_frame(self) -> OBDTelemetryFrame:
        if not self.is_connected():
            raise ConnectionError("Simulated OBD adapter is not connected. Call connect() first.")

        # Replay from dataset if available
        if self._df is not None and len(self._df) > 0:
            row = self._df.iloc[self._current_index % len(self._df)]
            self._current_index += 1

            dtcs = []
            if self._fault_injection:
                dtcs.append(self._fault_injection)
            elif pd.notna(row.get("TROUBLE_CODES")):
                dtcs.append(str(row["TROUBLE_CODES"]))

            return OBDTelemetryFrame(
                timestamp=datetime.datetime.now().isoformat(),
                engine_rpm=float(row.get("ENGINE_RPM", 1600.0)),
                speed_kmh=float(row.get("SPEED", 50.0)),
                coolant_temp_c=float(row.get("ENGINE_COOLANT_TEMP", 86.0)),
                engine_load_pct=float(row.get("ENGINE_LOAD", 35.0)),
                throttle_pos_pct=float(row.get("THROTTLE_POS", 18.0)),
                intake_air_temp_c=float(row.get("AIR_INTAKE_TEMP", 32.0)),
                battery_voltage=13.9,
                trouble_codes=dtcs,
                latency_ms=12.5,
                is_simulated=True
            )

        # Realistic dynamic driving cycle synthesis
        t = time.time()
        base_rpm = 1500.0 + 800.0 * np.sin(t * 0.1)
        base_speed = max(0.0, 45.0 + 35.0 * np.sin(t * 0.1 - 0.5))
        base_coolant = 88.0 + 4.0 * np.sin(t * 0.02)
        base_load = max(10.0, min(80.0, 35.0 + 25.0 * np.sin(t * 0.15)))
        base_throttle = max(5.0, min(60.0, 18.0 + 15.0 * np.sin(t * 0.15)))

        dtcs = [self._fault_injection] if self._fault_injection else []

        return OBDTelemetryFrame(
            timestamp=datetime.datetime.now().isoformat(),
            engine_rpm=round(base_rpm, 1),
            speed_kmh=round(base_speed, 1),
            coolant_temp_c=round(base_coolant, 1),
            engine_load_pct=round(base_load, 1),
            throttle_pos_pct=round(base_throttle, 1),
            intake_air_temp_c=34.0,
            battery_voltage=13.8,
            trouble_codes=dtcs,
            latency_ms=8.0,
            is_simulated=True
        )

    def get_trouble_codes(self) -> List[str]:
        frame = self.read_frame()
        return frame.trouble_codes

    def clear_trouble_codes(self) -> bool:
        self._fault_injection = None
        return True

    def stream_telemetry(self, interval_sec: float = 1.0, max_frames: Optional[int] = None) -> Generator[OBDTelemetryFrame, None, None]:
        """Generates continuous stream of vehicle telemetry frames."""
        count = 0
        while self.is_connected():
            yield self.read_frame()
            count += 1
            if max_frames and count >= max_frames:
                break
            time.sleep(interval_sec)


class ELM327SerialAdapter(BaseOBDAdapter):
    """
    Physical ELM327 OBD-II hardware adapter driver for Bluetooth, USB, or WiFi connections.
    Sends standard AT Hayes commands and SAE J1979 Mode 01 hex requests.
    """

    def __init__(self, port: str = "/dev/rfcomm0", baudrate: int = 38400, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self._status = OBDConnectionStatus.DISCONNECTED

    def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None, timeout: Optional[float] = None) -> bool:
        """
        Initializes serial/bluetooth connection to ELM327 OBD-II adapter:
        1. ATZ (Reset chip)
        2. ATE0 (Echo off)
        3. ATL0 (Linefeeds off)
        4. ATH0 (Headers off)
        5. ATSP0 (Auto-detect vehicle CAN/K-Line protocol)
        """
        if port:
            self.port = port
        if baudrate:
            self.baudrate = baudrate
        if timeout:
            self.timeout = timeout

        try:
            import serial
            self._status = OBDConnectionStatus.CONNECTING
            self.serial_conn = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
            
            # ELM327 initialization handshake
            self._send_raw("ATZ\r")
            time.sleep(0.5)
            self._send_raw("ATE0\r")
            self._send_raw("ATL0\r")
            self._send_raw("ATH0\r")
            self._send_raw("ATSP0\r")
            self._status = OBDConnectionStatus.CONNECTED
            return True
        except ImportError:
            self._status = OBDConnectionStatus.ERROR
            print("[ELM327Adapter] pyserial is not installed. To use physical hardware: pip install pyserial")
            return False
        except Exception as e:
            self._status = OBDConnectionStatus.ERROR
            print(f"[ELM327Adapter] Physical connection to {self.port} failed: {e}")
            print("Tip: If using Bluetooth ELM327, pair and bind first via: sudo rfcomm bind 0 <MAC_ADDR>")
            return False

    def disconnect(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self._status = OBDConnectionStatus.DISCONNECTED

    def is_connected(self) -> bool:
        return self._status == OBDConnectionStatus.CONNECTED and self.serial_conn and self.serial_conn.is_open

    def _send_raw(self, cmd: str) -> str:
        """Sends raw ASCII command to ELM327 and reads prompt response."""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise ConnectionError("Serial port not open.")
        self.serial_conn.write(cmd.encode("ascii"))
        response = self.serial_conn.read_until(b">").decode("ascii", errors="ignore")
        return response.strip().replace(">", "").strip()

    def query_pid(self, mode: str, pid: str) -> Optional[str]:
        """Queries standard OBD Mode and PID (e.g. Mode '01', PID '0C' for RPM)."""
        cmd = f"{mode}{pid}\r"
        res = self._send_raw(cmd)
        if "NO DATA" in res or "UNABLE TO CONNECT" in res or "ERROR" in res:
            return None
        return res

    def read_frame(self) -> OBDTelemetryFrame:
        """Reads all core vehicle sensors sequentially via standard Mode 01 PIDs."""
        start_t = time.time()
        
        # 010C: Engine RPM -> ((A * 256) + B) / 4
        rpm_res = self.query_pid("01", "0C")
        rpm = self._decode_rpm(rpm_res) if rpm_res else 0.0

        # 010D: Speed -> A (km/h)
        spd_res = self.query_pid("01", "0D")
        speed = self._decode_single_byte(spd_res) if spd_res else 0.0

        # 0105: Coolant Temp -> A - 40 (°C)
        cool_res = self.query_pid("01", "05")
        coolant = (self._decode_single_byte(cool_res) - 40.0) if cool_res else 85.0

        # 0104: Engine Load -> A * 100 / 255 (%)
        load_res = self.query_pid("01", "04")
        load = (self._decode_single_byte(load_res) * 100.0 / 255.0) if load_res else 30.0

        # 0111: Throttle Position -> A * 100 / 255 (%)
        throt_res = self.query_pid("01", "11")
        throttle = (self._decode_single_byte(throt_res) * 100.0 / 255.0) if throt_res else 15.0

        # 010F: Intake Air Temp -> A - 40 (°C)
        iat_res = self.query_pid("01", "0F")
        iat = (self._decode_single_byte(iat_res) - 40.0) if iat_res else 30.0

        latency = (time.time() - start_t) * 1000.0
        dtcs = self.get_trouble_codes()

        return OBDTelemetryFrame(
            timestamp=datetime.datetime.now().isoformat(),
            engine_rpm=rpm,
            speed_kmh=speed,
            coolant_temp_c=coolant,
            engine_load_pct=load,
            throttle_pos_pct=throttle,
            intake_air_temp_c=iat,
            trouble_codes=dtcs,
            latency_ms=round(latency, 1),
            is_simulated=False
        )

    def get_trouble_codes(self) -> List[str]:
        """Mode 03: Request Diagnostic Trouble Codes."""
        res = self._send_raw("03\r")
        if "NO DATA" in res or "43 00" in res:
            return []
        codes = []
        # Standard ELM327 returns bytes like: 43 01 33 03 00
        parts = res.split()
        if len(parts) >= 3 and parts[0] == "43":
            raw_hex = "".join(parts[1:])
            for i in range(0, len(raw_hex), 4):
                chunk = raw_hex[i:i+4]
                if len(chunk) == 4 and chunk != "0000":
                    prefix_char = {"0": "P0", "1": "P1", "4": "C0", "8": "B0", "C": "U0"}.get(chunk[0], "P0")
                    codes.append(f"{prefix_char}{chunk[1:]}")
        return codes

    def clear_trouble_codes(self) -> bool:
        """Mode 04: Clear DTCs and reset Malfunction Indicator Lamp."""
        res = self._send_raw("04\r")
        return "44" in res or "OK" in res

    def _decode_rpm(self, hex_str: str) -> float:
        try:
            tokens = hex_str.strip().split()
            a = int(tokens[-2], 16)
            b = int(tokens[-1], 16)
            return ((a * 256.0) + b) / 4.0
        except Exception:
            return 0.0

    def _decode_single_byte(self, hex_str: str) -> float:
        try:
            tokens = hex_str.strip().split()
            return float(int(tokens[-1], 16))
        except Exception:
            return 0.0


def create_obd_connection(mode: str = "simulated", port: Optional[str] = None) -> BaseOBDAdapter:
    """
    Factory function for OBD-II adapter connection.
    Pass mode='simulated' for testbed replay or mode='hardware' for real ELM327 Bluetooth/USB.
    """
    if mode.lower() == "hardware":
        adapter = ELM327SerialAdapter(port=port or "/dev/rfcomm0")
        success = adapter.connect()
        if not success:
            print("[OBDInterface] Physical adapter unavailable. Falling back to SimulatedOBDAdapter.")
            sim = SimulatedOBDAdapter(replay_csv_path=os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv"))
            sim.connect()
            return sim
        return adapter
    else:
        sim = SimulatedOBDAdapter(replay_csv_path=os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv"))
        sim.connect()
        return sim


if __name__ == "__main__":
    print("Testing OBD Hardware Abstraction Layer (Simulated Adapter):")
    obd = create_obd_connection(mode="simulated")
    print(f"Connected Status: {obd.is_connected()}")
    
    print("\nReading 3 Sample Telemetry Frames:")
    for i, frame in enumerate(obd.stream_telemetry(interval_sec=0.1, max_frames=3), 1):
        print(f"Frame #{i}: RPM={frame.engine_rpm} | Speed={frame.speed_kmh} km/h | Coolant={frame.coolant_temp_c}°C | Load={frame.engine_load_pct}% | DTC={frame.trouble_codes}")

    print("\nInjecting Fault DTC 'P0133' into simulator:")
    obd.inject_fault("P0133")
    fault_frame = obd.read_frame()
    print(f"Frame with Injected Fault: DTCs={fault_frame.trouble_codes}")
    obd.disconnect()
    print("OBD interface test complete.")
