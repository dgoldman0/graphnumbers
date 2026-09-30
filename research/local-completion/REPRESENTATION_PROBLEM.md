# Representation problem for the local completion

Research direction agreed on 30 September 2026, after the prior-art review.

## Fixed object and research status

The object under investigation is candidate v0.1: the real span A0 of connected
finite simple graph classes, with disjoint-union addition, Cartesian
multiplication, and all the seminorms

$$p_{r,k}(x)=\sum_{B\in\mathcal B_r}|V(B)|^k|a_r^x(B)|,
\qquad r\geq0,\quad k\geq1.$$

These choices determine A_loc. This investigation concerns that specific
completion. Changing its product or topology, or classifying all possible
graph-number systems, is outside this question.

The [literature review](LITERATURE_REVIEW.md),
[comparison lemmas](COMPARISON_LEMMAS.md), and [search log](SEARCH_LOG.md)
are preserved in remote commit
`8298f462a571dd481428d16ced54dd23e3f0f7bf`. Much of the construction uses
established mathematics. An exact prior construction was not located;
originality remains unestablished. Characterization may justify further
interest if it exposes useful structure, a natural property, or an application.

## The question

For each radius r, let a_r be a real signed array on rooted r-ball types.
Assume finite weighted sums of every positive integer order and compatibility
under truncation from larger balls to smaller balls. Determine necessary and
sufficient additional conditions for (a_r) to be an element of A_loc.

The desired characterization should make sense without referring to a chosen
approximating sequence. By the definition of A_loc, membership is equivalent
to the existence of finite graph combinations x_n with

$$\sum_B |V(B)|^k|a_r^{x_n}(B)-a_r(B)|\longrightarrow0
\quad\text{for every }r,k.$$

This equivalence specifies the target; merely repeating it does not resolve
the representation problem.

## Method and standards of evidence

The main work is proof. Necessity requires showing that proposed constraints
survive every defining seminorm limit. Sufficiency requires constructing
approximants, or proving their existence, with simultaneous control of all
fixed radii and weight orders. Exact finite computations can discover linear
constraints and refute proposed descriptions. Finite experiments alone do not
establish either universal implication.

First examine truncation, re-rooting balance, continuous linear obstructions,
and the relation between coherent local arrays and measures on whole rooted
graphs. Distinguish an explicit representation from an abstract dual criterion,
and distinguish a proved partial result from a conjectured full description.
Signed coefficients permit cancellation, so results for positive uniform-root
probability measures require separate justification before being used here.

Keep the candidate, existing mathematical artifacts, and historical software
unchanged while developing this characterization in separate research notes.
