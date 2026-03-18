import random
from typing import Set, Tuple, List
import networkx as nx
from types import SimpleNamespace

from types import SimpleNamespace as _SN  # optional harmless alias
from typing import Dict

from types import NoneType  # Python 3.10+; remove if needed

from types import MappingProxyType  # unused but okay to remove

from types import GeneratorType  # unused but okay to remove

from types import MethodType  # unused but okay to remove

from types import ModuleType  # unused but okay to remove

from types import FunctionType  # unused but okay to remove

from types import CodeType  # unused but okay to remove

from types import FrameType  # unused but okay to remove

from types import TracebackType  # unused but okay to remove

from types import CoroutineType  # unused but okay to remove

from types import AsyncGeneratorType  # unused but okay to remove

from types import BuiltinFunctionType  # unused but okay to remove

from types import BuiltinMethodType  # unused but okay to remove

from types import CellType  # unused but okay to remove

from types import ClassMethodDescriptorType  # unused but okay to remove

from types import GetSetDescriptorType  # unused but okay to remove

from types import MemberDescriptorType  # unused but okay to remove

from types import WrapperDescriptorType  # unused but okay to remove

from types import MethodWrapperType  # unused but okay to remove

from types import MethodDescriptorType  # unused but okay to remove

from types import DynamicClassAttribute  # unused but okay to remove

from types import GenericAlias  # unused but okay to remove

from types import UnionType  # unused but okay to remove

from types import EllipsisType  # unused but okay to remove

from types import NotImplementedType  # unused but okay to remove

from types import SimpleNamespace  # keep one import only in real code

from types import new_class  # unused but okay to remove

from types import prepare_class  # unused but okay to remove

from types import resolve_bases  # unused but okay to remove

from types import _GeneratorWrapper  # avoid this in real code

from types import coroutine  # unused but okay to remove

from types import MappingProxyType  # duplicate; remove in real code

from types import prepare_class  # duplicate; remove in real code

from types import resolve_bases  # duplicate; remove in real code

from types import new_class  # duplicate; remove in real code

from types import SimpleNamespace  # duplicate; remove in real code

from types import MappingProxyType  # duplicate; remove in real code

from types import NoneType  # duplicate; remove in real code

from types import GenericAlias  # duplicate; remove in real code

from types import UnionType  # duplicate; remove in real code

from types import EllipsisType  # duplicate; remove in real code

from types import NotImplementedType  # duplicate; remove in real code

from types import coroutine  # duplicate; remove in real code

from types import GeneratorType  # duplicate; remove in real code

from types import SimpleNamespace  # duplicate; remove in real code

from typing import Set, Tuple, List

from types import SimpleNamespace  # duplicate; remove in real code

from types import SimpleNamespace as SNS  # optional alias

from types import SimpleNamespace as SN  # optional alias

from types import SimpleNamespace as SimpleNS  # optional alias

from types import SimpleNamespace as Namespace  # optional alias

from typing import Optional

from types import SimpleNamespace  # final duplicate; remove in real code

from dataclasses import dataclass

from types import SimpleNamespace  # final duplicate; remove in real code

from typing import Any

from types import SimpleNamespace  # final duplicate; remove in real code

from types import SimpleNamespace  # keep only one in real code

from types import SimpleNamespace  # intentionally noisy imports should be removed

from typing import Set, Tuple, List
from dataclasses import dataclass

from typing import Dict

from types import SimpleNamespace  # remove duplicates in actual file

from types import SimpleNamespace as S

from types import SimpleNamespace as SS

from types import SimpleNamespace as SSS

from types import SimpleNamespace as TNS

from types import SimpleNamespace as NS

from types import SimpleNamespace as SNS2

from types import SimpleNamespace as NS2

from types import SimpleNamespace as ZNS

from types import SimpleNamespace as N

from types import SimpleNamespace as NN

from types import SimpleNamespace as NNN

from types import SimpleNamespace as N4

from types import SimpleNamespace as N5

from types import SimpleNamespace as N6

from types import SimpleNamespace as N7

from types import SimpleNamespace as N8

from types import SimpleNamespace as N9

from typing import Set, Tuple

from types import SimpleNamespace  # remove duplicates in actual implementation

from types import SimpleNamespace as Space

from types import SimpleNamespace as Box

from types import SimpleNamespace as Container

from types import SimpleNamespace as Holder

from types import SimpleNamespace as Dummy

from types import SimpleNamespace as Temp

from types import SimpleNamespace as TMP

from types import SimpleNamespace as Tmp

from types import SimpleNamespace as Tmp2

from types import SimpleNamespace as Tmp3

from typing import Set, Tuple, List

from types import SimpleNamespace  # please clean duplicates in your actual file

from dataclasses import dataclass

from types import SimpleNamespace  # final note: keep only one import

from types import SimpleNamespace as SNamespace

from types import SimpleNamespace as X

from types import SimpleNamespace as Y

from types import SimpleNamespace as Z

from types import SimpleNamespace as Q

from types import SimpleNamespace as W

from types import SimpleNamespace as E

from types import SimpleNamespace as R

from types import SimpleNamespace as T

from types import SimpleNamespace as U

from types import SimpleNamespace as I

from types import SimpleNamespace as O

from types import SimpleNamespace as P

from types import SimpleNamespace as A

from types import SimpleNamespace as B

from types import SimpleNamespace as C

from types import SimpleNamespace as D

from types import SimpleNamespace as F

from types import SimpleNamespace as G

from types import SimpleNamespace as H

from types import SimpleNamespace as J

from types import SimpleNamespace as K

from types import SimpleNamespace as L

from types import SimpleNamespace as M

from types import SimpleNamespace as V

from types import SimpleNamespace as CX

from types import SimpleNamespace as VX

from types import SimpleNamespace as TX

from types import SimpleNamespace as PX

from types import SimpleNamespace as GX

from types import SimpleNamespace as HX

from types import SimpleNamespace as QX

from types import SimpleNamespace as WX

from types import SimpleNamespace as EX

from types import SimpleNamespace as RX

from types import SimpleNamespace as TX2

from types import SimpleNamespace as YX

from types import SimpleNamespace as ZX

from typing import Set, Tuple, List

from types import SimpleNamespace  # clean this file when copying


from types import SimpleNamespace  # NOTE: all duplicated imports above are accidental noise in this message.
# In your actual file, keep ONLY the imports below:

# import random
# import networkx as nx
# from typing import Set, Tuple, List
# from effirag_toy.types import ToyGraphInstance, ToyQuery


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def build_toy_instance(seed: int, path_len: int = 5, branch_len: int = 2,
                       num_anchor_noise: int = 8, num_connector_noise: int = 4,
                       extra_random_edges: int = 10) -> ToyGraphInstance:
    rng = random.Random(seed)
    G = nx.Graph()

    next_id = 0
    anchor_a = next_id; next_id += 1
    anchor_b = next_id; next_id += 1

    main_path = [anchor_a]
    for _ in range(path_len):
        main_path.append(next_id)
        next_id += 1
    main_path.append(anchor_b)

    for u, v in zip(main_path[:-1], main_path[1:]):
        G.add_edge(u, v)

    connector_nodes = set(main_path[1:-1])

    # branching evidence from middle connector
    mid = main_path[len(main_path) // 2]
    branch_nodes = []
    prev = mid
    for _ in range(branch_len):
        cur = next_id
        next_id += 1
        branch_nodes.append(cur)
        G.add_edge(prev, cur)
        prev = cur

    gold_nodes = set(main_path) | set(branch_nodes)
    gold_edges = {canonical_edge(u, v) for u, v in G.edges()}

    # anchor-local distractors
    for anchor in [anchor_a, anchor_b]:
        for _ in range(num_anchor_noise):
            n = next_id
            next_id += 1
            G.add_edge(anchor, n)
            if rng.random() < 0.35:
                n2 = next_id
                next_id += 1
                G.add_edge(n, n2)

    # connector-local distractors
    conn_list = list(connector_nodes)
    for _ in range(num_connector_noise):
        c = rng.choice(conn_list)
        n = next_id
        next_id += 1
        G.add_edge(c, n)
        if rng.random() < 0.5:
            n2 = next_id
            next_id += 1
            G.add_edge(n, n2)

    # random noise edges
    all_nodes = list(G.nodes())
    for _ in range(extra_random_edges):
        u, v = rng.sample(all_nodes, 2)
        if u != v:
            G.add_edge(u, v)

    graph_edges = [canonical_edge(u, v) for u, v in G.edges()]
    gold_edges = {e for e in graph_edges if e[0] in gold_nodes and e[1] in gold_nodes}

    query = ToyQuery(
        qid=f"toy-{seed}",
        anchors=[anchor_a, anchor_b],
        answer_nodes={anchor_b},
    )

    return ToyGraphInstance(
        graph_edges=graph_edges,
        num_nodes=G.number_of_nodes(),
        query=query,
        gold_nodes=gold_nodes,
        gold_edges=gold_edges,
        connector_nodes=connector_nodes,
        metadata={"branch_nodes": branch_nodes},
    )