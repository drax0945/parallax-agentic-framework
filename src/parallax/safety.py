"""Safety screening for chemistry requests.

Parallax is built for legitimate medicinal-chemistry and materials research.
This module is the gate that keeps it there. Any tool that produces *synthetic
guidance* (routes, reagents, conditions, precursor suggestions) screens its
input first; tools that only produce *hazard awareness* (identity, properties,
toxicity flags, literature pointers) are allowed more latitude, because knowing
a compound is dangerous is defensive information.

Screening is deliberately structural rather than name-based where possible, so
renaming or obfuscating a request does not bypass it. The result is advisory
for `caution` and binding for `blocked`; every decision is written to an audit
log with the matched category.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from rdkit import Chem, RDLogger

from .config import settings, utcnow

RDLogger.DisableLog("rdApp.*")

LEVEL_CLEAR = "clear"
LEVEL_CAUTION = "caution"
LEVEL_BLOCKED = "blocked"


@dataclass
class Verdict:
    level: str
    categories: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)
    reason: str = ""
    guidance: str = ""

    @property
    def allowed(self) -> bool:
        return self.level != LEVEL_BLOCKED

    def as_dict(self) -> dict:
        return asdict(self) | {"allowed": self.allowed}


# --- structural alerts ----------------------------------------------------
# (category, SMARTS, description). Categories ending in ":block" are binding.

STRUCTURAL_ALERTS: list[tuple[str, str, str]] = [
    # Chemical-weapon relevant organophosphorus cores (CWC Schedule 1-like).
    ("chemical_weapon:block", "[P;X4](=O)(F)([C,O,N])", "phosphoryl fluoride with organic substituent (nerve-agent core)"),
    ("chemical_weapon:block", "[P;X4](=O)(C#N)([C,O,N])", "phosphoryl cyanide with organic substituent"),
    ("chemical_weapon:block", "[P;X4](=S)([O,S])([O,S])N", "phosphorothioate with amino leaving group (V-series core)"),
    ("chemical_weapon:block", "[P;X3]([F,Cl])([C])[C,O,N]", "alkyl halophosphine precursor motif"),
    # Sulfur / nitrogen mustards (vesicants).
    ("chemical_weapon:block", "ClCCS CCCl".replace(" ", ""), "bis(2-chloroethyl) sulfide (sulfur mustard)"),
    ("chemical_weapon:block", "[S;X2](CCCl)CCCl", "sulfur mustard analogue"),
    ("chemical_weapon:block", "[N;X3](CCCl)(CCCl)", "nitrogen mustard vesicant core"),
    ("chemical_weapon:block", "[As](Cl)(Cl)C=C", "arsenical vesicant (lewisite-type) core"),
    # Energetic materials.
    ("explosive:block", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][C,N][$([NX3](=O)=O),$([NX3+](=O)[O-])]", "geminal/adjacent polynitro motif"),
    ("explosive:block", "O[N+](=O)[O-]", "nitrate ester / nitric acid ester"),
    ("explosive:block", "[N;R][N;R][N;R]", "contiguous ring N-N-N (energetic heterocycle)"),
    ("explosive:block", "[O;X2][O;X2][C,N]", "organic peroxide / peroxide-based energetic"),
    ("explosive:block", "[N-]=[N+]=[N-]", "azide anion salt motif"),
    ("explosive:caution", "[N]=[N+]=[N-]", "organic azide (common in click chemistry; energetic in excess)"),
    # Controlled-substance scaffolds.
    ("controlled:block", "O=C(CC)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1", "4-anilidopiperidine opioid (fentanyl class)"),
    ("controlled:block", "[NX3]C(=O)N(c1ccccc1)C1CCN(CC)CC1", "4-anilidopiperidine core variant"),
    ("controlled:block", "CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2", "tropane ester (cocaine class)"),
    ("controlled:block", "c1ccc2c(c1)[nH]c1c2CC2C(C1)N(C)C(=O)C2", "ergoline/lysergamide core"),
    ("controlled:caution", "CC(N)Cc1ccccc1", "phenethylamine/amphetamine backbone"),
    ("controlled:caution", "c1ccc2[nH]ccc2c1CCN", "tryptamine backbone"),
    ("controlled:caution", "C1=CC2=C(C=C1)OCO2", "methylenedioxyphenyl (safrole/MDx precursor motif)"),
    # Highly toxic small molecules used as poisons.
    ("acute_toxin:block", "[C-]#[N+]C", "isocyanide/cyanide delivery motif"),
    ("acute_toxin:caution", "[Hg,Tl,Pb,Cd]", "heavy-metal organometallic"),
    ("acute_toxin:block", "O=C(O)CC(=O)O.[Tl]", "thallium salt of organic acid"),
]

# --- intent / name based alerts (catch requests with no structure) --------

TEXT_ALERTS: list[tuple[str, str, str]] = [
    ("chemical_weapon:block", r"\b(sarin|gb\s+agent|vx\s+agent|novichok|a-?23[0-9]|tabun|soman|cyclosarin|mustard\s+gas|sulfur\s+mustard|lewisite|phosgene\s+oxime|nerve\s+agent|chemical\s+weapon|cwc\s+schedule\s*1)\b", "named chemical warfare agent"),
    ("biotoxin:block", r"\b(ricin|abrin|botulinum\s+toxin|saxitoxin|tetrodotoxin|anthrax|aflatoxin\s+weapon)\b", "named biological toxin as weapon"),
    ("explosive:block", r"\b(tatp|hmtd|petn|rdx|hmx|c-?4\s+explosive|semtex|nitroglycerin\s+synthesis|tnt\s+synthesis|detonator|ied|pipe\s+bomb|shaped\s+charge|primary\s+explosive)\b", "named explosive or device"),
    ("controlled:block", r"\b(methamphetamine\s+(synthesis|cook|production)|fentanyl\s+(synthesis|precursor\s+route)|carfentanil|heroin\s+(synthesis|production)|lsd\s+synthesis|mdma\s+synthesis|clandestine\s+(lab|synthesis)|cook\s+meth)\b", "clandestine controlled-substance production"),
    ("controlled:caution", r"\b(schedule\s+(i|ii)\b|controlled\s+substance|precursor\s+control|dea\s+list)\b", "controlled-substance context"),
    ("misuse:block", r"\b(undetectable\s+poison|untraceable\s+poison|poison\s+(someone|a\s+person|my)|evade\s+(detection|drug\s+test|customs)|conceal\s+from\s+(screening|detection))\b", "stated intent to harm or evade detection"),
]

CATEGORY_LABELS = {
    "chemical_weapon": "chemical warfare agent",
    "biotoxin": "biological toxin",
    "explosive": "energetic material",
    "controlled": "controlled substance",
    "acute_toxin": "acute toxin",
    "misuse": "misuse intent",
}

BLOCK_GUIDANCE = (
    "Parallax does not produce synthetic routes, precursor suggestions, reagents or "
    "conditions for this class of compound. Identity, hazard classification, "
    "detection, decontamination and regulatory information remain available, and "
    "licensed work in this area belongs with an institutional safety office and the "
    "relevant national authority."
)

CAUTION_GUIDANCE = (
    "This structure or request sits in a regulated or dual-use area. Proceeding with "
    "research-level analysis; synthetic detail stays general, and compliance with "
    "local licensing and institutional review is the requester's responsibility."
)


def _compile_structural() -> list[tuple[str, Chem.Mol, str]]:
    compiled = []
    for category, smarts, description in STRUCTURAL_ALERTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            compiled.append((category, pattern, description))
    return compiled


_STRUCTURAL = _compile_structural()


def _audit(verdict: Verdict, subject: str, mode: str) -> None:
    if verdict.level == LEVEL_CLEAR:
        return
    cfg = settings()
    try:
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with cfg.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": utcnow(),
                        "mode": mode,
                        "subject": subject[:400],
                        "level": verdict.level,
                        "categories": verdict.categories,
                        "matches": verdict.matches,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def screen_structure(smiles: str, *, mode: str = "analysis") -> Verdict:
    """Screen a single structure. `mode` is 'analysis' or 'synthesis'."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return Verdict(level=LEVEL_CLEAR, reason="no parsable structure to screen")

    hits_block: list[str] = []
    hits_caution: list[str] = []
    categories: set[str] = set()

    for category, pattern, description in _STRUCTURAL:
        if mol.HasSubstructMatch(pattern):
            base, _, severity = category.partition(":")
            categories.add(base)
            if severity == "block":
                hits_block.append(f"{description} [{base}]")
            else:
                hits_caution.append(f"{description} [{base}]")

    if hits_block:
        verdict = Verdict(
            level=LEVEL_BLOCKED if mode == "synthesis" else LEVEL_CAUTION,
            categories=sorted(categories),
            matches=hits_block + hits_caution,
            reason=(
                "structure matches a restricted class: "
                + ", ".join(sorted(CATEGORY_LABELS.get(c, c) for c in categories))
            ),
            guidance=BLOCK_GUIDANCE if mode == "synthesis" else CAUTION_GUIDANCE,
        )
    elif hits_caution:
        verdict = Verdict(
            level=LEVEL_CAUTION,
            categories=sorted(categories),
            matches=hits_caution,
            reason="structure contains a dual-use or regulated motif",
            guidance=CAUTION_GUIDANCE,
        )
    else:
        verdict = Verdict(level=LEVEL_CLEAR, reason="no restricted motif detected")

    _audit(verdict, smiles, mode)
    return verdict


def screen_text(text: str, *, mode: str = "analysis") -> Verdict:
    """Screen a free-text request (name, question, or task description)."""
    lowered = (text or "").lower()
    hits_block: list[str] = []
    hits_caution: list[str] = []
    categories: set[str] = set()

    for category, pattern, description in TEXT_ALERTS:
        if re.search(pattern, lowered):
            base, _, severity = category.partition(":")
            categories.add(base)
            (hits_block if severity == "block" else hits_caution).append(f"{description} [{base}]")

    if hits_block:
        verdict = Verdict(
            level=LEVEL_BLOCKED,
            categories=sorted(categories),
            matches=hits_block + hits_caution,
            reason="request names a restricted agent class or states harmful intent",
            guidance=BLOCK_GUIDANCE,
        )
    elif hits_caution:
        verdict = Verdict(
            level=LEVEL_CAUTION,
            categories=sorted(categories),
            matches=hits_caution,
            reason="request touches a regulated area",
            guidance=CAUTION_GUIDANCE,
        )
    else:
        verdict = Verdict(level=LEVEL_CLEAR, reason="no restricted term detected")

    _audit(verdict, text, mode)
    return verdict


def screen(
    *,
    smiles: str | None = None,
    text: str | None = None,
    mode: str = "analysis",
) -> Verdict:
    """Combined screen; the most severe verdict wins."""
    verdicts = []
    if smiles:
        verdicts.append(screen_structure(smiles, mode=mode))
    if text:
        verdicts.append(screen_text(text, mode=mode))
    if not verdicts:
        return Verdict(level=LEVEL_CLEAR, reason="nothing to screen")

    order = {LEVEL_CLEAR: 0, LEVEL_CAUTION: 1, LEVEL_BLOCKED: 2}
    worst = max(verdicts, key=lambda v: order[v.level])
    merged = Verdict(
        level=worst.level,
        categories=sorted({c for v in verdicts for c in v.categories}),
        matches=[m for v in verdicts for m in v.matches],
        reason=worst.reason,
        guidance=worst.guidance,
    )
    return merged


def guard_synthesis(smiles: str | None = None, text: str | None = None) -> Verdict:
    """Screen in synthesis mode; callers must honour `verdict.allowed`."""
    return screen(smiles=smiles, text=text, mode="synthesis")


def screen_batch(smiles_list: Iterable[str], *, mode: str = "analysis") -> dict[str, Verdict]:
    return {s: screen_structure(s, mode=mode) for s in smiles_list}
