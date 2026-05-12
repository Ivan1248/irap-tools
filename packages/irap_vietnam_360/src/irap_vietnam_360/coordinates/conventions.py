from typing import Literal
import numpy as np


class Frame(str):
    """A 3-character string representing a coordinate convention with named axes.
    
    Attributes:
        AXES: A dictionary of named axes ordered in pairs (positive, negative),
            e.g. {"U": "Up", "D": "Down", "L": "Left", "R": "Right"}
    """

    AXES = dict()

    def __new__(cls, value: str | list | tuple) -> "Frame":
        if isinstance(value, (list, tuple)):
            value = ''.join(str(c) for c in value)
        elif not isinstance(value, str):
            value = str(value)
        if len(value) != 3 or not all(c in cls.AXES for c in value):
            raise ValueError(
                f"Convention must be defined by 3 characters of {set(cls.AXES)}, got: {value}")
        return super().__new__(cls, value)

    def __instancecheck__(cls, value) -> bool:
        """Checks if a string is a valid coordinate convention."""
        if not isinstance(value, str):
            return False
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def multiply_axis(cls, axis: str, sign: Literal[1, -1]) -> str:
        """Multiplies the axis by the sign."""
        axes = list(cls.AXES.keys())
        return axes[axes.index(axis) ^ int(sign < 0)]  # even <-> odd

    @classmethod
    def negate_axis(cls, axis: str) -> str:
        """Returns the opposite axis."""
        return cls.multiply_axis(axis, -1)

    def get_axis(self, axis: int | str, sign: Literal[1, -1]) -> str:
        """Get the axis character based on index and sign."""
        return type(self).multiply_axis(self[axis] if isinstance(axis, int) else axis, sign)

    def get_conversion_string(self, target_frame: "Frame") -> "RelativeFrame":
        """Get the axis mapping from this frame to `target_frame`.

        Returns a 3-character `RelativeFrame` built from the set {X, Y, Z, x, y, z}:
        - Uppercase (X/Y/Z) denotes a direct (positive) use of the corresponding
          source axis (first/second/third axis).
        - Lowercase (x/y/z) denotes the same axis but with inverted sign.

        The i-th character specifies which source axis maps to the i-th axis of
        `target_frame`, including whether it must be inverted.

        For example, if this frame is "FLU" and the target frame is "RDF", the
        conversion string is "yzX".
        """
        assert isinstance(target_frame, type(self))
        return RelativeFrame([
            "XYZ"[self.index(axis)] if axis in self else "xyz"[self.index(self.negate_axis(axis))]
            for axis in target_frame
        ])

    def convert(self, conversion_string: "RelativeFrame") -> "Frame":
        """Convert the frame to the target frame.
        
        For example, if this frame is "FLU" and the conversion string is "yzX", the
        resulting frame is "RDF".
        """
        indexes, signs = RelativeFrame(conversion_string).to_indexes_and_signs()
        return type(self)([self.multiply_axis(self[i], sign) for i, sign in zip(indexes, signs)])

    def convert_coordinates(self, coordinates, target_frame: "Frame", array_factory=np.array):
        """Convert the coordinates to the target frame.
        
        Args:
            coordinates: array with shape (..., 3)
            target_frame (str | Frame): The target frame to convert to.
        
        Returns:
            The converted coordinates.
        """
        conversion_string = self.get_conversion_string(target_frame) if isinstance(
            target_frame, type(self)) else target_frame
        indexes, signs = conversion_string.to_indexes_and_signs()
        return coordinates[..., indexes] * array_factory(signs)


class RelativeFrame(Frame):
    """A 3-character string representing relative coordinate convention."""

    AXES = {
        "X": "Positive first axis",  # 0
        "x": "Negative first axis",
        "Y": "Positive second axis",  # 1
        "y": "Negative second axis",
        "Z": "Positive third axis",  # 2
        "z": "Negative third axis",
    }

    @classmethod
    def from_indexes_and_signs(cls, indexes: list[int],
                               signs: list[Literal[1, -1]]) -> "RelativeFrame":
        """Create a relative frame from indexes and signs."""
        return cls([("XYZ" if s > 0 else "xyz")[i] for i, s in zip(indexes, signs)])

    def to_indexes_and_signs(self) -> tuple[list[int], list[Literal[1, -1]]]:
        """Get the indexes and signs of the conversion string."""
        indexes = ["XYZ".index(c.upper()) for c in self]
        signs = [1 if c.isupper() else -1 for c in self]
        return indexes, signs


class BodyFrame(Frame):
    """A 3-character string representing body coordinate convention."""

    AXES = {
        "F": "Forward",  # anterior
        "B": "Backward",  # posterior
        "L": "Left", 
        "R": "Right", 
        "U": "Up",  # dorsal, zenith
        "D": "Down",  # ventral, nadir
    }
    ROLL_AXIS = "F"
    PITCH_AXIS = "L"
    YAW_AXIS = "U"


class WorldFrame(Frame):
    """A 3-character string representing world coordinate convention."""

    AXES = {
        "N": "North",
        "S": "South",
        "W": "West",
        "E": "East",
        "U": "Up",
        "D": "Down",
    }


BodyAxis = Literal[*tuple(BodyFrame.AXES.keys())]
WorldAxis = Literal[*tuple(WorldFrame.AXES.keys())]
RelativeAxis = Literal[*tuple(RelativeFrame.AXES.keys())]

POSITIVE_GRAVITY_AXIS: WorldAxis = "U"  # Gravity should be positive
ZERO_ROLL_LEFT: WorldAxis = "N"
ZERO_PITCH_UP: WorldAxis = "U"
ZERO_YAW_FORWARD: WorldAxis = "E"

# Common conventions as constants

# References:
# https://www.ros.org/reps/rep-0103.html


class BodyConventions:
    ROSBody = BodyFrame("FLU")  # ROS REP-103 body frame
    Optical = BodyFrame("RDF")  # Optical, camera, GyroFlow camera frame
    PhonePortrait = BodyFrame("RUB")  # Smartpone portrait (on bottom edge)
    PhoneLandscape = BodyFrame("ULB")  # Smartpone landscape (on left edge)
    Insta360X4 = BodyFrame("BDR")  # Insta360 X4 IMU frame
    GyroFlow = BodyFrame("URF")  # GyroFlow body frame?


class WorldConventions:
    ENU = WorldFrame("ENU")  # East-North-Up, geographic, ROS REP-103 world frame
    NED = WorldFrame("NED")  # North-East-Down, aerospace
    NWU = WorldFrame("NWU")  # North-West-Up
