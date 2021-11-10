# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "эй би собака эн ди точка ру" } -> "эй би собака эн ди точка ру"

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit_no_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        ).optimize() | pynini.cross("1", "eins")
        graph_zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv"))).optimize()
        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        def add_space_after_char():
            return pynini.closure(NEMO_NOT_QUOTE - pynini.accep(" ") + insert_space) + (
                NEMO_NOT_QUOTE - pynini.accep(" ")
            )

        user_name = pynutil.delete("username: \"") + add_space_after_char() + pynutil.delete("\"")

        convert_defaults = NEMO_NOT_QUOTE | domain_common | server_common
        domain = convert_defaults + pynini.closure(insert_space + convert_defaults)
        domain @= pynini.cdrewrite(graph_symbols | graph_digit, "", "", NEMO_SIGMA)

        domain = pynutil.delete("domain: \"") + domain + pynutil.delete("\"")
        protocol = (
            pynutil.delete("protocol: \"")
            + add_space_after_char() @ pynini.cdrewrite(graph_symbols, "", "", NEMO_SIGMA)
            + pynutil.delete("\"")
        )
        graph = (pynini.closure(protocol + pynini.accep(" "), 0, 1) + domain) | (
            user_name + pynini.accep(" ") + pynutil.insert("at ") + domain
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
