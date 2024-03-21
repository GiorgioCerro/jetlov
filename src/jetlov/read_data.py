# This file is part of LundNet by F. Dreyer and H. Qu
# adapted from code written by G. Salam

from abc import ABC, abstractmethod
from math import pow

# ======================================================================
from pathlib import Path
from typing import Dict, Iterator, List

import fastjet as fj
from heparchy.read.hdf import HdfReader

EventType = List[Dict[str, float]]


def heparchy_adapter(file: Path, shower_info: bool = False) -> Iterator[EventType]:
    # in_dir = Path(in_dir)
    # for fp in filter(lambda p: p.suffix in {".h5", ".hdf5"}, in_dir.iterdir()):
    with HdfReader(file) as hep_file:
        try:
            proc = hep_file["background"]
        except KeyError:
            proc = hep_file["signal"]
        for event in proc:
            pmu = event.pmu
            pmu.dtype.names = ("px", "py", "pz", "E")
            if shower_info:
                yield event.custom["shower_id"][0]
            else:
                yield list(
                    map(lambda row: dict(zip(row.dtype.names, map(float, row))), pmu)
                )


# class Reader:
#    def __init__(self, in_dir: Path, nmax: int = -1) -> None:
#        self.in_dir = in_dir
#        self.nmax = nmax
#        self.reset()
#
#    def reset(self) -> None:
#        self._iter = enumerate(heparchy_adapter(self.in_dir))
#
#    def __iter__(self) -> Iterator[EventType]:
#        return self
#
#    def __next__(self) -> EventType:
#        n, event = next(self._iter)
#        if n == self.nmax:
#            print("# Exiting after having read nmax jet declusterings")
#            raise StopIteration
#        return event
#
#    def next_event(self) -> EventType:
#        return next(self)


# ======================================================================
# class Image(ABC):
#    """Image which transforms point-like information into pixelated 2D
#    images which can be processed by convolutional neural networks."""
#
#    def __init__(self, infile, nmax):
#        self.reader = Reader(infile, nmax)
#
#    # ----------------------------------------------------------------------
#    @abstractmethod
#    def process(self, event):
#        pass
#
#    # ----------------------------------------------------------------------
#    def __iter__(self):
#        # needed for iteration to work
#        return self
#
#    # ----------------------------------------------------------------------
#    def __next__(self):
#        ev = self.reader.next_event()
#        if (ev is None):
#            raise StopIteration
#        else:
#            return self.process(ev)
#
#    # ----------------------------------------------------------------------
#    def next(self): return self.__next__()
#
#    # ----------------------------------------------------------------------
#    def values(self):
#        res = []
#        while True:
#            event = self.reader.next_event()
#            if event != None:
#                res.append(self.process(event))
#            else:
#                break
#        self.reader.reset()
#        return res


class Reader:
    def __init__(self, in_dir: Path, nmax: int = -1) -> None:
        self.in_dir = in_dir
        self.nmax = nmax
        self.reset()

    def reset(self) -> None:
        self._iter = enumerate(heparchy_adapter(self.in_dir))

    def __iter__(self) -> Iterator[EventType]:
        return self

    def __next__(self) -> EventType:
        n, event = next(self._iter)
        if n == self.nmax:
            print("# Exiting after having read nmax jet declusterings")
            raise StopIteration
        return event

    def next_event(self) -> EventType:
        return next(self)


# ======================================================================
class Image(ABC):
    """Image which transforms point-like information into pixelated 2D
    images which can be processed by convolutional neural networks."""

    def __init__(self, infile, nmax):
        self.reader = Reader(infile, nmax)

    # ----------------------------------------------------------------------
    @abstractmethod
    def process(self, event):
        pass

    # ----------------------------------------------------------------------
    def __iter__(self):
        # needed for iteration to work
        return self

    # ----------------------------------------------------------------------
    def __next__(self):
        ev = self.reader.next_event()
        if ev is None:
            raise StopIteration
        else:
            return self.process(ev)

    # ----------------------------------------------------------------------
    def next(self):
        return self.__next__()

    # ----------------------------------------------------------------------
    def values(self):
        res = []
        while True:
            event = self.reader.next_event()
            if event is not None:
                res.append(self.process(event))
            else:
                break
        self.reader.reset()
        return res


# ======================================================================
class Jets(Image):
    """Read input file with jet constituents and transform into python jets."""

    # ----------------------------------------------------------------------
    def __init__(
        self, infile, nmax, pseudojets=True, groomer=None, algorithm="cambridge"
    ):
        Image.__init__(self, infile, nmax)
        if algorithm == "cambridge":
            self.jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
        elif algorithm == "kt":
            self.jet_def = fj.JetDefinition(fj.kt_algorithm, 1000.0)
        elif algorithm == "antikt":
            self.jet_def = fj.JetDefinition(fj.antikt_algorithm, 1000.0)
        else:
            import warnings

            warnings.warn("Algorithm not found. Going with the default: cambridge")
            self.jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.0)
        self.pseudojets = pseudojets
        self.groomer = groomer

    # ----------------------------------------------------------------------
    def process(self, event):
        constits = []
        if self.pseudojets or self.groomer:
            for p in event[1:]:
                constits.append(fj.PseudoJet(p["px"], p["py"], p["pz"], p["E"]))
            jets = self.jet_def(constits)
            if len(jets) > 0:
                if self.groomer:
                    constits = self.groomer(jets[0], self.pseudojets)
                    return self.jet_def(constits)[0] if self.pseudojets else constits
                return jets[0]
            return fj.PseudoJet()
        else:
            for p in event[1:]:
                constits.append([p["px"], p["py"], p["pz"], p["E"]])
            return constits


# ======================================================================
class GroomJetRSD:
    """Recursive Soft Drop groomer applicable on fastjet PseudoJets"""

    # ----------------------------------------------------------------------
    def __init__(self, zcut=0.05, beta=1.0, R0=1.0):
        """Initialize RSD with its parameters."""
        self.zcut = zcut
        self.beta = beta
        self.R0 = R0

    def __call__(self, jet, pseudojets=True):
        constits = []
        self._groom(jet, constits, pseudojets)
        return constits

    def _groom(self, j, constits, pseudojets):
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        if j.has_parents(j1, j2):
            # order the parents in pt
            if j2.pt() > j1.pt():
                j1, j2 = j2, j1
            delta = j1.delta_R(j2)
            z = j2.pt() / (j1.pt() + j2.pt())
            remove_soft = z < self.zcut * pow(delta / self.R0, self.beta)
            if remove_soft:
                self._groom(j1, constits, pseudojets)
            else:
                self._groom(j1, constits, pseudojets)
                self._groom(j2, constits, pseudojets)
        else:
            if pseudojets:
                constits.append(fj.PseudoJet(j.px(), j.py(), j.pz(), j.E()))
            else:
                constits.append([j.px(), j.py(), j.pz(), j.E()])
