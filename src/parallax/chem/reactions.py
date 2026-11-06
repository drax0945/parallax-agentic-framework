"""Reaction template library for retrosynthetic analysis.

Each template is a retro-direction reaction SMARTS (product >> precursors)
plus the practical knowledge a chemist would bring to it: typical conditions,
what it tolerates, where it fails, how reliable it is at scale, and a
literature query that will find real precedents.

Two things are kept apart on purpose:

* the **template prior** (`reliability`) — a coarse, hand-set expectation of
  how often this transform works, used only to rank proposals;
* the **precedent** — actual reported examples, which must be looked up with
  the literature tools. A proposed step with no precedent found is reported as
  unprecedented, not as plausible.

That separation is what stops a route from looking authoritative when it is
really just pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


@dataclass
class Template:
    name: str
    retro_smarts: str
    forward_description: str
    reaction_class: str
    conditions: str
    reliability: float  # 0-1 prior on "this works as drawn"
    tolerates: str = ""
    fails_when: str = ""
    citation: str | None = None  # key in provenance.SOFTWARE_CITATIONS-like store
    doi: str | None = None
    reference_title: str | None = None
    reference_authors: list[str] = field(default_factory=list)
    reference_year: int | None = None
    reference_venue: str | None = None
    precedent_query: str = ""
    scale_notes: str = ""
    macrocyclization: bool = False
    procedural: str | None = None  # ring-opening handled by retro.py bond surgery
    _reaction: Any = field(default=None, repr=False, compare=False)

    def reaction(self) -> Any:
        if self._reaction is None:
            self._reaction = AllChem.ReactionFromSmarts(self.retro_smarts)
            if self._reaction is not None:
                self._reaction.Initialize()
        return self._reaction


TEMPLATES: list[Template] = [
    # --- amide / peptide bonds -------------------------------------------
    Template(
        name="Amide coupling (acid + amine)",
        retro_smarts="[C:1](=[O:2])[NX3;H1,H2:3]>>[C:1](=[O:2])[OX2H1].[N:3]",
        forward_description="Carboxylic acid + amine -> amide, via an activated ester",
        reaction_class="C-N bond formation",
        conditions="HATU or EDC/HOBt, DIPEA, DMF or DCM, 0-25 °C; T3P or propylphosphonic anhydride for scale",
        reliability=0.92,
        tolerates="most heterocycles, halides, esters, Boc/Cbz groups",
        fails_when="the amine is very hindered or strongly deactivated (anilines with ortho-EWG), or a free carboxylate competes",
        doi="10.1016/j.tet.2005.08.031",
        reference_title="Amino acid coupling chemistry for peptide synthesis",
        reference_authors=["Montalbetti, Christian A. G. N.", "Falque, Virginie"],
        reference_year=2005,
        reference_venue="Tetrahedron",
        precedent_query="amide coupling HATU {product_class}",
        scale_notes="Workhorse step; coupling-agent choice drives cost and purification burden at scale.",
    ),
    Template(
        name="Macrolactamization",
        retro_smarts="[C;R:1](=[O:2])[NX3;R;H0,H1:3]>>[C:1](=[O:2])[OX2H1].[N:3]",
        forward_description="Intramolecular amide formation closing a macrocyclic ring",
        reaction_class="macrocyclization",
        conditions="High dilution (1-5 mM), HATU/DIPEA or T3P, slow addition; pseudo-dilution on resin is an alternative",
        reliability=0.55,
        tolerates="pre-organised chains with a turn-inducing element",
        fails_when="the linear precursor has no conformational bias — oligomerisation dominates",
        doi="10.1021/cr0509388",
        reference_title="Macrolactonization and macrolactamization",
        reference_authors=["Parenty, Alexis", "Moreau, Xavier", "Campagne, Jean-Marc"],
        reference_year=2006,
        reference_venue="Chemical Reviews",
        precedent_query="macrolactamization high dilution macrocycle",
        scale_notes="High dilution means large solvent volumes; often the cost-determining step.",
        macrocyclization=True,
        procedural="amide",
    ),
    # --- C-C couplings ----------------------------------------------------
    Template(
        name="Suzuki-Miyaura coupling",
        retro_smarts="[c:1]-!@[c:2]>>[c:1][Br].[c:2][B]([OX2H1])[OX2H1]",
        forward_description="Aryl halide + aryl boronic acid -> biaryl, Pd-catalysed",
        reaction_class="C-C bond formation",
        conditions="Pd(dppf)Cl2 or XPhos-Pd-G3, K2CO3 or K3PO4, dioxane/water, 80-100 °C",
        reliability=0.9,
        tolerates="amines, amides, nitriles, free NH heterocycles (with the right ligand)",
        fails_when="2-heteroaryl boronic acids protodeboronate; very hindered ortho,ortho'-substituted pairs stall",
        doi="10.1021/cr00039a007",
        reference_title="Palladium-catalyzed cross-coupling reactions of organoboron compounds",
        reference_authors=["Miyaura, Norio", "Suzuki, Akira"],
        reference_year=1995,
        reference_venue="Chemical Reviews",
        precedent_query="Suzuki coupling boronic acid {product_class}",
        scale_notes="Best-developed coupling for manufacture; boronate stability is the main risk.",
    ),
    Template(
        name="Suzuki coupling (boronate ester variant)",
        retro_smarts="[c:1]-!@[c:2]>>[c:1][Br].[c:2][B]1[OX2][C]([CH3])([CH3])[C]([CH3])([CH3])[OX2]1",
        forward_description="Aryl halide + pinacol boronate -> biaryl",
        reaction_class="C-C bond formation",
        conditions="Pd(dppf)Cl2, K2CO3, dioxane/water 85 °C",
        reliability=0.88,
        tolerates="substrates where the free boronic acid is unstable",
        fails_when="slow transmetalation with very electron-rich partners",
        doi="10.1021/cr00039a007",
        reference_title="Palladium-catalyzed cross-coupling reactions of organoboron compounds",
        reference_authors=["Miyaura, Norio", "Suzuki, Akira"],
        reference_year=1995,
        reference_venue="Chemical Reviews",
        precedent_query="pinacol boronate Suzuki coupling heteroaryl",
    ),
    Template(
        name="Sonogashira coupling",
        retro_smarts="[c:1]-!@[CX2:2]#[CX2:3]>>[c:1][I].[CX2:2]#[CX2:3]",
        forward_description="Aryl halide + terminal alkyne -> aryl alkyne",
        reaction_class="C-C bond formation",
        conditions="Pd(PPh3)4, CuI, Et3N, THF or DMF, 25-60 °C",
        reliability=0.85,
        tolerates="most heterocycles; copper-free variants exist for Cu-sensitive substrates",
        fails_when="Glaser homocoupling competes under air; free NH2 can coordinate",
        doi="10.1021/cr050992x",
        reference_title="The Sonogashira reaction: a booming methodology in synthetic organic chemistry",
        reference_authors=["Chinchilla, Rafael", "Nájera, Carmen"],
        reference_year=2007,
        reference_venue="Chemical Reviews",
        precedent_query="Sonogashira coupling aryl iodide terminal alkyne",
    ),
    Template(
        name="Heck coupling",
        retro_smarts="[c:1]-!@[CX3:2]=[CX3:3]>>[c:1][Br].[CX3:2]=[CX3:3]",
        forward_description="Aryl halide + alkene -> aryl alkene",
        reaction_class="C-C bond formation",
        conditions="Pd(OAc)2, P(o-tol)3, Et3N, DMF, 100 °C",
        reliability=0.72,
        tolerates="esters, nitriles, ketones",
        fails_when="regio- and stereoselectivity are substrate-dependent; internal alkenes give mixtures",
        precedent_query="Heck reaction aryl halide alkene regioselectivity",
    ),
    Template(
        name="Negishi / Kumada alkyl-aryl coupling",
        retro_smarts="[c:1]-!@[CX4;H2:2]>>[c:1][Br].[CX4;H2:2][Zn][Br]",
        forward_description="Aryl halide + alkylzinc (or Grignard) -> alkyl arene",
        reaction_class="C-C bond formation",
        conditions="Pd(dppf)Cl2 or NiCl2(dppe), THF, 0-60 °C",
        reliability=0.65,
        tolerates="halides on the ring if chemoselectivity is managed",
        fails_when="protic or electrophilic functional groups are present — organozinc is basic/nucleophilic",
        precedent_query="Negishi coupling alkylzinc aryl bromide",
    ),
    # --- C-N bonds --------------------------------------------------------
    Template(
        name="Buchwald-Hartwig amination",
        retro_smarts="[c:1]-!@[NX3;!$(N=*);!$([NX3][CX3]=[OX1,SX1]);!$([NX3][SX4](=O)=O):2]>>[c:1][Br].[N:2]",
        forward_description="Aryl halide + amine -> aryl amine, Pd/ligand catalysed",
        reaction_class="C-N bond formation",
        conditions="Pd2(dba)3 or BrettPhos/RuPhos precatalyst, NaOtBu or Cs2CO3, toluene or dioxane, 80-110 °C",
        reliability=0.82,
        tolerates="primary and secondary amines, anilines, many azoles with ligand tuning",
        fails_when="strong base is incompatible with esters/epimerisable centres; free OH/COOH poison the catalyst",
        doi="10.1021/acs.chemrev.6b00512",
        reference_title="Applications of palladium-catalyzed C-N cross-coupling reactions",
        reference_authors=["Ruiz-Castillo, Paula", "Buchwald, Stephen L."],
        reference_year=2016,
        reference_venue="Chemical Reviews",
        precedent_query="Buchwald-Hartwig amination aryl bromide ligand",
        scale_notes="Ligand and Pd removal (to <10 ppm) are the practical constraints.",
    ),
    Template(
        name="SNAr amination",
        retro_smarts="[c;$(c[n,$(c[F,Cl,Br])]),$(c[CX3]=[OX1]),$(c[N+](=O)[O-]),$(c[CX2]#[NX1]):1]-!@[NX3;H1,H2:2]>>[c:1][Cl].[N:2]",
        forward_description="Electron-poor (hetero)aryl halide + amine -> aryl amine, no metal",
        reaction_class="C-N bond formation",
        conditions="DIPEA or K2CO3, DMSO/NMP, 60-140 °C (or microwave)",
        reliability=0.86,
        tolerates="most things — no catalyst to poison; the cleanest option when the ring allows it",
        fails_when="the ring is not activated; 2-/4-substitution relative to the ring N is usually required",
        doi="10.1002/anie.201902216",
        reference_title="Concerted nucleophilic aromatic substitution reactions",
        reference_authors=["Rohrbach, Simon", "Smith, Andrew J.", "Pang, Jia Hao", "Poole, Darren L.", "Tuttle, Tell", "Chiba, Shunsuke", "Murphy, John A."],
        reference_year=2019,
        reference_venue="Angewandte Chemie International Edition",
        precedent_query="SNAr chloropyridine amine DIPEA DMSO",
        scale_notes="Preferred over Pd coupling at scale when feasible — no metal removal needed.",
    ),
    Template(
        name="Reductive amination",
        retro_smarts="[CX4;H1,H2;!$(C[OX2H1]);!$(C=O):1]-!@[NX3;!$([NX3][CX3]=[OX1]);!$(N=*):2]>>[C:1]=[OX1].[N:2]",
        forward_description="Aldehyde or ketone + amine -> secondary/tertiary amine",
        reaction_class="C-N bond formation",
        conditions="NaBH(OAc)3 in DCE, or NaBH3CN/MeOH with AcOH; H2 and Pd/C for scale",
        reliability=0.88,
        tolerates="esters, nitriles, halides, Boc groups",
        fails_when="hindered ketones with weakly nucleophilic anilines; over-alkylation with primary amines",
        doi="10.1021/jo960057x",
        reference_title="Reductive amination of aldehydes and ketones with sodium triacetoxyborohydride",
        reference_authors=["Abdel-Magid, Ahmed F.", "Carson, Kenneth G.", "Harris, Bruce D.", "Maryanoff, Cynthia A.", "Shah, Rekha D."],
        reference_year=1996,
        reference_venue="Journal of Organic Chemistry",
        precedent_query="reductive amination sodium triacetoxyborohydride",
    ),
    Template(
        name="N-alkylation with alkyl halide",
        retro_smarts="[NX3;!$(N=*);!$([NX3][CX3]=[OX1]):1]-!@[CX4;H2:2]>>[N:1].[C:2][Br]",
        forward_description="Amine or azole NH + alkyl halide -> N-alkyl product",
        reaction_class="C-N bond formation",
        conditions="K2CO3 or Cs2CO3, DMF or acetonitrile, 25-80 °C",
        reliability=0.78,
        tolerates="most; simple and cheap",
        fails_when="regiochemistry on unsymmetrical azoles is unreliable; over-alkylation and quaternisation occur",
        precedent_query="N-alkylation azole alkyl bromide cesium carbonate regioselectivity",
    ),
    Template(
        name="Sulfonamide formation",
        retro_smarts="[SX4:1](=[OX1:2])(=[OX1:3])[NX3:4]>>[S:1](=[O:2])(=[O:3])[Cl].[N:4]",
        forward_description="Sulfonyl chloride + amine -> sulfonamide",
        reaction_class="C-N bond formation",
        conditions="Pyridine or Et3N, DCM, 0-25 °C",
        reliability=0.9,
        tolerates="most functionality",
        fails_when="sulfonyl chloride hydrolysis competes with poorly nucleophilic amines",
        precedent_query="sulfonyl chloride amine sulfonamide pyridine",
    ),
    Template(
        name="Urea formation (isocyanate route)",
        retro_smarts="[NX3;H1,H2:1][CX3:2](=[OX1:3])[NX3;H1,H2:4]>>[N:1].[N:4][C:2]=[O:3]",
        forward_description="Amine + isocyanate -> urea",
        reaction_class="C-N bond formation",
        conditions="DCM or THF, 0-25 °C; or CDI/triphosgene when the isocyanate is unavailable",
        reliability=0.85,
        tolerates="most; very clean when the isocyanate is accessible",
        fails_when="isocyanates are moisture-sensitive and sometimes not commercial",
        precedent_query="urea formation amine isocyanate CDI",
    ),
    Template(
        name="Boc protection",
        retro_smarts="[NX3:1][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]>>[N:1]",
        forward_description="Amine + Boc2O -> Boc carbamate (protection)",
        reaction_class="protecting group",
        conditions="Boc2O, Et3N or NaHCO3, THF/water or DCM, 25 °C",
        reliability=0.95,
        tolerates="nearly everything",
        fails_when="rarely fails; removal needs acid, which must be tolerated downstream",
        precedent_query="Boc protection amine Boc2O",
    ),
    # --- C-O bonds --------------------------------------------------------
    Template(
        name="Williamson ether synthesis",
        retro_smarts="[#6:1][OX2:2]-!@[CX4;H1,H2:3]>>[#6:1][O:2][H].[C:3][Br]",
        forward_description="Alcohol/phenol + alkyl halide -> ether",
        reaction_class="C-O bond formation",
        conditions="K2CO3 or Cs2CO3, DMF or acetone, 25-80 °C; NaH/THF for poorly acidic alcohols",
        reliability=0.8,
        tolerates="esters (with mild base), halides, nitriles",
        fails_when="secondary/tertiary halides eliminate instead of substituting",
        precedent_query="Williamson ether synthesis phenol alkyl bromide potassium carbonate",
    ),
    Template(
        name="Mitsunobu etherification",
        retro_smarts="[c:1][OX2:2]-!@[CX4;H1:3]>>[c:1][O:2][H].[C:3][OX2H1]",
        forward_description="Phenol + secondary alcohol -> aryl ether with inversion of configuration",
        reaction_class="C-O bond formation",
        conditions="PPh3, DIAD or DEAD, THF, 0-25 °C",
        reliability=0.7,
        tolerates="acid-sensitive groups; the standard way to set a secondary benzylic ether stereocentre",
        fails_when="phenol pKa above ~11 is unreactive; triphenylphosphine oxide removal is tedious",
        doi="10.1021/cr800278z",
        reference_title="Mitsunobu and related reactions: advances and applications",
        reference_authors=["Swamy, K. C. Kumara", "Kumar, N. N. Bhuvan", "Balaraman, E.", "Kumar, K. V. P. Pavan"],
        reference_year=2009,
        reference_venue="Chemical Reviews",
        precedent_query="Mitsunobu etherification phenol secondary alcohol inversion",
        scale_notes="Stoichiometric phosphine and hydrazine by-products make this expensive at scale.",
    ),
    Template(
        name="Macroetherification (SNAr or Williamson closure)",
        retro_smarts="[c;R:1][OX2;R:2][CX4;R;H1,H2:3]>>[c:1][OX2H1].[C:3][Br]",
        forward_description="Intramolecular C-O bond formation closing a macrocycle",
        reaction_class="macrocyclization",
        conditions="Cs2CO3, DMF or NMP, high dilution (1-10 mM), 60-100 °C",
        reliability=0.5,
        tolerates="electron-poor arenes (SNAr-assisted closure is markedly easier)",
        fails_when="the tether is too short to reach, or the phenol is insufficiently nucleophilic",
        doi="10.1039/c4sc03089c",
        reference_title="Macrocycles: lessons from the distant past, recent developments, and future directions",
        reference_authors=["Yudin, Andrei K."],
        reference_year=2015,
        reference_venue="Chemical Science",
        precedent_query="macrocyclization intramolecular etherification cesium carbonate high dilution",
        macrocyclization=True,
        procedural="ether",
    ),
    Template(
        name="Ester formation",
        retro_smarts="[CX3:1](=[OX1:2])[OX2:3][#6:4]>>[C:1](=[O:2])[OX2H1].[O:3][#6:4]",
        forward_description="Carboxylic acid + alcohol -> ester",
        reaction_class="C-O bond formation",
        conditions="EDC/DMAP, or acid chloride + base; Fischer esterification (H2SO4/MeOH) for methyl esters",
        reliability=0.88,
        tolerates="most functionality under the carbodiimide variant",
        fails_when="tertiary alcohols and very hindered acids need the Yamaguchi or Mukaiyama protocol",
        precedent_query="Steglich esterification EDC DMAP",
    ),
    Template(
        name="Macrolactonization",
        retro_smarts="[CX3;R:1](=[OX1:2])[OX2;R:3][CX4;R:4]>>[C:1](=[O:2])[OX2H1].[O:3][C:4]",
        forward_description="Intramolecular esterification closing a macrolactone",
        reaction_class="macrocyclization",
        conditions="Yamaguchi (2,4,6-trichlorobenzoyl chloride/DMAP) or Mukaiyama, high dilution, slow addition",
        reliability=0.55,
        tolerates="complex polyketide-type chains — the classic natural-product endgame",
        fails_when="without conformational pre-organisation, dimer/oligomer dominates",
        doi="10.1021/cr0509388",
        reference_title="Macrolactonization and macrolactamization",
        reference_authors=["Parenty, Alexis", "Moreau, Xavier", "Campagne, Jean-Marc"],
        reference_year=2006,
        reference_venue="Chemical Reviews",
        precedent_query="Yamaguchi macrolactonization high dilution",
        macrocyclization=True,
        procedural="ester",
    ),
    Template(
        name="Buchwald C-O coupling (aryl ether)",
        retro_smarts="[c:1]-!@[OX2:2][#6:3]>>[c:1][Br].[O:2][#6:3]",
        forward_description="Aryl halide + alcohol -> aryl ether, Pd or Cu catalysed",
        reaction_class="C-O bond formation",
        conditions="RockPhos-Pd-G3 or CuI/phenanthroline, Cs2CO3, toluene, 90-110 °C",
        reliability=0.6,
        tolerates="primary and secondary alcohols",
        fails_when="β-hydride elimination and competing C-N coupling when amines are present",
        precedent_query="palladium catalysed C-O coupling aryl halide alcohol RockPhos",
    ),
    # --- ring formation ---------------------------------------------------
    Template(
        name="Ring-closing metathesis",
        retro_smarts="[CX3;R:1]=[CX3;R:2]>>[C:1]=[CH2].[C:2]=[CH2]",
        forward_description="Diene -> cycloalkene, Ru-catalysed, releasing ethylene",
        reaction_class="macrocyclization",
        conditions="Grubbs II or Hoveyda-Grubbs, DCM or toluene, 40-80 °C, high dilution",
        reliability=0.62,
        tolerates="amides, esters, many heterocycles; the standard macrocyclisation when alkenes are allowed",
        fails_when="basic amines poison Ru; E/Z selectivity needs control; the alkene usually has to be reduced afterwards",
        doi="10.1016/j.tet.2004.05.055",
        reference_title="Olefin metathesis",
        reference_authors=["Grubbs, Robert H."],
        reference_year=2004,
        reference_venue="Tetrahedron",
        precedent_query="ring closing metathesis macrocycle Grubbs high dilution",
        macrocyclization=True,
        procedural="alkene",
    ),
    Template(
        name="CuAAC click triazole formation",
        retro_smarts="[cX3:1]1[cX3:2][nX3:3][nX2:4][nX2:5]1>>[C:1]#[C:2].[N:3]=[N+:4]=[N-:5]",
        forward_description="Azide + terminal alkyne -> 1,4-triazole, Cu(I) catalysed",
        reaction_class="ring formation",
        conditions="CuSO4/sodium ascorbate, t-BuOH/water, 25 °C",
        reliability=0.9,
        tolerates="essentially everything — bioorthogonal",
        fails_when="Cu removal matters for biology; organic azides are energetic and need handling care",
        precedent_query="CuAAC click chemistry triazole azide alkyne",
    ),
    Template(
        name="Pyrazole from 1,3-dicarbonyl",
        retro_smarts="[cX3:1]1[cX3:2][cX3:3][nX3:4][nX2:5]1>>[C:1](=O)[C:2][C:3]=O.[N:4][N:5]",
        forward_description="1,3-Diketone + hydrazine -> pyrazole",
        reaction_class="ring formation",
        conditions="EtOH, AcOH catalytic, 60-80 °C",
        reliability=0.75,
        tolerates="aryl and alkyl substituents",
        fails_when="unsymmetrical diketones give regioisomer mixtures with substituted hydrazines",
        precedent_query="pyrazole synthesis 1,3-diketone hydrazine regioselectivity",
    ),
    Template(
        name="Amide to 1,3,4-oxadiazole / cyclodehydration",
        retro_smarts="[cX3:1]1[oX2:2][cX3:3][nX2:4][nX2:5]1>>[C:1](=O)[N:5][N:4][C:3]=O",
        forward_description="Diacylhydrazide -> 1,3,4-oxadiazole by cyclodehydration",
        reaction_class="ring formation",
        conditions="Burgess reagent, TsCl/Et3N, or POCl3, 60-110 °C",
        reliability=0.7,
        tolerates="aryl and heteroaryl",
        fails_when="harsh dehydration conditions damage sensitive groups",
        precedent_query="1,3,4-oxadiazole cyclodehydration diacylhydrazide",
    ),
    # --- functional group interconversion ---------------------------------
    Template(
        name="Nitro reduction to aniline",
        retro_smarts="[c:1][NX3;H2:2]>>[c:1][N+](=[OX1])[O-]",
        forward_description="Nitroarene -> aniline",
        reaction_class="reduction",
        conditions="H2, Pd/C, EtOH; or Fe/NH4Cl, or SnCl2 when alkenes/halides must survive",
        reliability=0.9,
        tolerates="most; choose the reductant to preserve halides and alkenes",
        fails_when="hydrogenolysis of benzylic groups and dehalogenation under Pd/H2",
        precedent_query="nitro reduction aniline iron ammonium chloride",
    ),
    Template(
        name="Nitrile from aryl halide (cyanation)",
        retro_smarts="[c:1][CX2]#[NX1]>>[c:1][Br]",
        forward_description="Aryl halide + cyanide source -> aryl nitrile",
        reaction_class="C-C bond formation",
        conditions="Zn(CN)2 or K4[Fe(CN)6], Pd2(dba)3/dppf, DMF, 100-120 °C",
        reliability=0.75,
        tolerates="esters, amides, many heterocycles",
        fails_when="cyanide poisons Pd if released too fast — the ferrocyanide source exists for exactly this",
        precedent_query="palladium catalysed cyanation aryl bromide zinc cyanide",
        scale_notes="Cyanide handling and quench are the dominant safety and permitting consideration.",
    ),
    Template(
        name="Ester hydrolysis to acid",
        retro_smarts="[CX3:1](=[OX1:2])[OX2H1]>>[C:1](=[O:2])[OX2][CH3]",
        forward_description="Methyl ester -> carboxylic acid (saponification)",
        reaction_class="deprotection",
        conditions="LiOH or NaOH, THF/MeOH/water, 25-60 °C",
        reliability=0.93,
        tolerates="most; the standard way to unmask an acid before coupling",
        fails_when="base-labile stereocentres epimerise; competing amide hydrolysis in forcing conditions",
        precedent_query="ester hydrolysis lithium hydroxide carboxylic acid",
    ),
    Template(
        name="Grignard addition to carbonyl",
        retro_smarts="[CX4:1]([OX2H1:2])[#6:3]>>[C:1]=[O:2].[#6:3][Br]",
        forward_description="Aldehyde/ketone + organomagnesium -> secondary/tertiary alcohol",
        reaction_class="C-C bond formation",
        conditions="RMgBr, THF or Et2O, -78 to 0 °C",
        reliability=0.7,
        tolerates="ethers, acetals, alkenes",
        fails_when="protic groups, esters, nitriles and nitro groups are incompatible",
        precedent_query="Grignard addition aldehyde secondary alcohol",
    ),
    Template(
        name="Alcohol oxidation to ketone/aldehyde",
        retro_smarts="[CX3:1]=[OX1:2]>>[C:1][OX2H1:2]",
        forward_description="Alcohol -> aldehyde or ketone",
        reaction_class="oxidation",
        conditions="Dess-Martin periodinane or Swern; TEMPO/NaOCl or Mn-catalysed aerobic oxidation at scale",
        reliability=0.85,
        tolerates="most; over-oxidation to acid is the main risk with aldehydes",
        fails_when="electron-rich heterocycles and sulfides are oxidised competitively",
        precedent_query="Dess-Martin oxidation alcohol aldehyde",
    ),
    Template(
        name="Alkene hydrogenation",
        retro_smarts="[CX4;H1,H2:1][CX4;H1,H2:2]>>[C:1]=[C:2]",
        forward_description="Alkene -> alkane (typically after RCM)",
        reaction_class="reduction",
        conditions="H2 (1-4 bar), Pd/C or PtO2, EtOAc or EtOH; Crabtree for directed diastereoselection",
        reliability=0.86,
        tolerates="amides, esters, most heterocycles",
        fails_when="benzylic deprotection and dehalogenation occur under Pd/H2; diastereoselectivity needs a directing group",
        precedent_query="alkene hydrogenation palladium macrocycle diastereoselectivity",
    ),
]


def compiled_templates() -> list[Template]:
    """SMARTS-driven templates whose reaction compiles (checked at first use)."""
    return [t for t in TEMPLATES if not t.procedural and t.reaction() is not None]


def procedural_templates() -> list[Template]:
    """Ring-opening templates applied by explicit bond surgery in `retro`."""
    return [t for t in TEMPLATES if t.procedural]


def template_by_name(name: str) -> Template | None:
    for template in TEMPLATES:
        if template.name.lower() == name.lower():
            return template
    return None


def library_summary() -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "class": t.reaction_class,
            "reliability_prior": t.reliability,
            "conditions": t.conditions,
            "macrocyclization": t.macrocyclization,
            "has_literature_reference": bool(t.doi),
        }
        for t in compiled_templates()
    ]


def register_template_citation(template: Template) -> str | None:
    """Register a template's literature reference in the run's citation registry."""
    if not template.doi:
        return None
    from ..provenance import Citation, registry

    return registry().add(
        Citation(
            key=f"doi:{template.doi.lower()}",
            kind="method",
            title=template.reference_title or template.name,
            source_db="parallax/reaction-template-library",
            authors=template.reference_authors,
            year=template.reference_year,
            venue=template.reference_venue,
            doi=template.doi,
            note=f"methodology reference for the '{template.name}' transform",
        )
    )


def validate_reaction_smiles(reaction_smiles: str) -> dict[str, Any]:
    """Check a reaction for atom balance and parsability.

    Catches the most common failure in machine-written chemistry: a reaction
    that looks right but does not conserve atoms.
    """
    try:
        reaction = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
    except Exception as exc:
        return {"valid": False, "error": f"unparsable reaction SMILES: {exc}"}
    if reaction is None:
        return {"valid": False, "error": "unparsable reaction SMILES"}

    def composition(block: str) -> tuple[dict[str, int], int]:
        """Element counts for one side of the reaction, written as SMILES."""
        counts: dict[str, int] = {}
        parsed = 0
        for component in filter(None, block.split(".")):
            mol = Chem.MolFromSmiles(component)
            if mol is None:
                continue
            parsed += 1
            for atom in Chem.AddHs(mol).GetAtoms():
                counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        return counts, parsed

    # Count atoms from the SMILES text rather than from reaction templates, whose
    # implicit valences are not calculated and cannot be hydrogenated.
    pieces = reaction_smiles.split(">")
    if len(pieces) == 3:
        reactant_block, agent_block, product_block = pieces
    elif len(pieces) == 2:
        reactant_block, agent_block, product_block = pieces[0], "", pieces[1]
    else:
        return {"valid": False, "error": "reaction SMILES must contain '>>' or 'a>b>c'"}

    left, n_reactants = composition(reactant_block)
    right, n_products = composition(product_block)
    _, n_agents = composition(agent_block)

    reactants = [None] * n_reactants
    products = [None] * n_products
    agents = [None] * n_agents
    elements = sorted(set(left) | set(right))
    imbalance = {
        element: right.get(element, 0) - left.get(element, 0)
        for element in elements
        if right.get(element, 0) != left.get(element, 0)
    }

    return {
        "valid": True,
        "n_reactants": len(reactants),
        "n_products": len(products),
        "n_agents": len(agents),
        "atom_balanced": not imbalance,
        "imbalance_product_minus_reactant": imbalance,
        "note": (
            "Atom-balanced as written."
            if not imbalance
            else "Not atom-balanced. Small imbalances are often fine (leaving groups, "
            "catalysts, water written implicitly), but check that the missing atoms are "
            "accounted for by reagents you intend to list."
        ),
    }
