"""Module for text watermarking algorithms."""

import re

from bitarray import bitarray
from bitarray.util import ba2hex, ba2int, int2ba
from langchain_core.runnables import RunnableLambda
from siphash import siphash24


class WatermarkError(Exception):
    """Base class for watermarking errors."""

    pass


class TextTooShortError(WatermarkError):
    """Raised when the text is too short to embed the watermark."""

    pass


class TextWatermark:
    """Base class for text watermarks."""

    def __init__(self):
        pass

    def apply(self, text: str) -> str:
        """Apply the watermarking algorithm to the given text.

        Parameters
        ----------
        text : str
            The text to be watermarked.

        Returns
        -------
        str
            The watermarked text.
        """
        raise NotImplementedError

    def remove(self, text: str) -> str:
        """Remove the watermark from the given text.

        Parameters
        ----------
        text : str
            The text to be cleaned.

        Returns
        -------
        str
            The cleaned text.
        """
        raise NotImplementedError

    def check(self, text: str) -> bool:
        """Check if the given text contains a watermark.

        Parameters
        ----------
        text : str
            The text to be checked.

        Returns
        -------
        bool
            True if the text contains a watermark, False otherwise.
        """
        raise NotImplementedError

    def apply_as_runnable(self) -> RunnableLambda:
        """Return the `apply` function as a Runnable."""
        return RunnableLambda(lambda text: self.apply(text))

    def check_as_runnable(self) -> RunnableLambda:
        """Return the `check` function as a Runnable."""
        return RunnableLambda(lambda text: self.check(text))


class TokenWatermark(TextWatermark):
    """Naive token-based watermarking algorithm.

    Parameters
    ----------
    key : str
        The secret key used to generate the token.
    format : str
        Python format-string to define the token's decoration. Defaults to "[{}]".
    front : bool
        Whether to put the token at the front of the text or at the back of the text.
        Defaults to True.
    """

    def __init__(self, key: bytes, fmt: str = "[{}]", front: bool = True):
        self.key = key
        self.fmt = fmt
        self.front = front

    def apply(self, text: str) -> str:
        """Apply the watermarking algorithm to the given text.

        Parameters
        ----------
        text : str
            The text to be watermarked.

        Returns
        -------
        str
            The watermarked text.
        """
        token = ba2hex(self._fingerprint(text))
        return (
            self.fmt.format(token) + text
            if self.front
            else text + self.fmt.format(token)
        )

    def remove(self, text: str) -> str:
        """Remove the watermark from the given text.

        Parameters
        ----------
        text : str
            The text to be cleaned.

        Returns
        -------
        str
            The cleaned text.
        """
        before, after = self.fmt.format(":").split(":")
        reg = r"([0-9a-f]+)"  # the token is an hexadecimal string
        pattern = f"{re.escape(before)}{reg}{re.escape(after)}"
        return re.sub(pattern, "", text)

    def check(self, text: str) -> bool:
        """Check if the given text contains a valid watermark.

        Parameters
        ----------
        text : str
            The text to be checked.

        Returns
        -------
        bool
            True if the text contains the watermark, False otherwise.
        """
        orig = self.remove(text)
        watermark = ba2hex(self._fingerprint(orig))

        before, after = self.fmt.format(":").split(":")
        reg = r"([0-9a-f]+)"  # the token is an hexadecimal string
        pattern = f"{re.escape(before)}{reg}{re.escape(after)}"

        match = re.search(pattern, text)
        if match:
            token = match.group(1)
            return token == watermark
        return False

    def _fingerprint(self, text: str) -> bitarray:
        """Generate a SipHash fingerprint of the given text."""
        sip = siphash24(self.key)
        sip.update(text.encode())
        watermark = bitarray()
        watermark.frombytes(sip.digest())
        return watermark


class Rizzo2016(TextWatermark):
    """Implementation of the watermarking algorithm proposed by Rizzo et al. in 2016.

    The algorithm has two steps:
    1. Generate a SipHash fingerprint of the text, using a secret key.
    2. Embed the fingerprint in the text by replacing characters based on a substitution table.

    Attributes
    ----------
    confusable_chars : dict[str, str]
        A dictionary mapping characters to their confusable counterparts.
    confusable_spaces : dict[int, str]
        A dictionary mapping 3-bit integers to their confusable space characters.

    Parameters
    ----------
    key : str
        The secret key used to generate the fingerprint.
    """

    confusable_chars = {
        "\u002d": "\u2010",  # -
        "\u003b": "\u037e",  # ;
        "\u0043": "\u216d",  # C
        "\u0044": "\u216e",  # D
        "\u004b": "\u212a",  # K
        "\u004c": "\u216c",  # L
        "\u004d": "\u216f",  # M
        "\u0056": "\u2164",  # V
        "\u0058": "\u2169",  # X
        "\u0063": "\u217d",  # c
        "\u0064": "\u217e",  # d
        "\u0069": "\u2170",  # i
        "\u006a": "\u0458",  # j
        "\u006c": "\u217c",  # l
        "\u0076": "\u2174",  # v
        "\u0078": "\u2179",  # x
    }

    confusable_spaces = {
        0b000: "\u0020",  # space
        0b001: "\u2000",  # >en quad
        0b010: "\u2004",  # three-per-em space
        0b011: "\u2005",  # four-per-em space
        0b100: "\u2008",  # punctuation space
        0b101: "\u2009",  # thin space
        0b110: "\u202f",  # narrow no-break space
        0b111: "\u205f",  # medium mathematical space
    }

    def __init__(self, key: bytes):
        super().__init__()
        if not isinstance(key, bytes):
            raise TypeError("The key must be a bytes object.")
        self.key = key

    def apply(self, text: str) -> str:
        """Apply the watermarking algorithm to the given text.

        Parameters
        ----------
        text : str
            The text to be watermarked.

        Returns
        -------
        str
            The watermarked text.
        """
        watermark = self._fingerprint(text)
        confusable = list(self.confusable_chars.keys()) + [" "]
        out = ""

        for c in text:
            if c in confusable and len(watermark) > 0:
                if c == " ":
                    bits = watermark[:3]
                    watermark = watermark[3:]
                    c = self.confusable_spaces[ba2int(bits)]
                else:
                    bit: bool = watermark.pop(0)
                    if bit:
                        c = self.confusable_chars[c]
            out += c

        if len(watermark) > 0:
            raise TextTooShortError(f"Text too short: {len(watermark)} bits left.")

        return out

    def remove(self, text: str) -> str:
        """Remove the watermark from the given text.

        Parameters
        ----------
        text : str
            The text to be cleaned.

        Returns
        -------
        str
            The cleaned text.
        """
        out = ""
        inv_char_map = {v: k for k, v in self.confusable_chars.items()}

        for c in text:
            if c in inv_char_map:
                c = inv_char_map[c]
            elif c in self.confusable_spaces.values():
                c = " "
            out += c

        return out

    def check(self, text: str) -> bool:
        """Check if the given text contains a valid watermark.

        Parameters
        ----------
        text : str
            The text to be checked.

        Returns
        -------
        bool
            True if the text contains the watermark, False otherwise.
        """
        watermark = bitarray()
        inv_chars = {v: k for k, v in self.confusable_chars.items()}
        inv_spaces = {v: k for k, v in self.confusable_spaces.items()}

        for c in text:
            if c in inv_chars:
                watermark.append(True)
            elif c in self.confusable_chars:
                watermark.append(False)
            elif c in inv_spaces:
                watermark.extend(int2ba(inv_spaces[c], 3))

            if len(watermark) >= 64:
                break

        orig = self.remove(text)

        return self._fingerprint(orig) == watermark

    def _fingerprint(self, text: str) -> bitarray:
        """Generate a SipHash fingerprint of the given text."""
        sip = siphash24(self.key)
        sip.update(text.encode())
        watermark = bitarray()
        watermark.frombytes(sip.digest())
        return watermark
