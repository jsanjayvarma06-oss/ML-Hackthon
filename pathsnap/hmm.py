"""HMM + Viterbi route reconstruction. OWNER: Sanjay.

Hidden state at GPS point i = one of its candidate arcs.
Emission  = classifier probability for that candidate.
Transition = high if same arc or connected (arcs[a,1] == arcs[b,0]); otherwise based on
             network distance vs GPS distance. Use a small floor, never 0.
Work in log space. Split the trip at large time gaps and decode each part separately.
"""


def viterbi(candidates, emission_logp, trip, gap_split_s=60.0):
    """Best candidate arc per GPS point.

    candidates: list of int arrays (per point), emission_logp: list of float arrays
    with the same shapes. Returns an int array of arc ids (-1 where no candidate).
    """
    raise NotImplementedError("TODO(Sanjay)")
