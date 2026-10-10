# ORBIT-all method sketch

This document defines the current optimizer hypotheses. It is deliberately explicit about which constructions are exact local sensitivities and which are practical proxies.

## Base update

For ordinary hidden matrices, ORBIT-all starts from the same Muon-style matrix candidate direction. A role-specific geometry may then reshape that candidate before the parameter step is applied. Family-level Frobenius restoration preserves the aggregate update norm so that the main ablations test geometry rather than an accidental learning-rate increase.

Parameters that are not major matrix families (biases, RMSNorm scales, ordinary scalars) remain on the auxiliary AdamW path.

## 1. Q/K score geometry

For a query-key score

```text
s = q^T R k
```

a query perturbation obeys

```text
delta s = delta q^T R k
```

and therefore

```text
E[(delta s)^2] = delta q^T R E[k k^T] R^T delta q.
```

This motivates opposite-side second moments. ORBIT-all retains ORBIT's optimizer-side rotary transport and 2x2 frequency-local inverse-metric conditioning.

For GQA, a KV head is shared by a contiguous group of query heads. The query metric uses the shared K statistics, while the K metric uses the mean second moment of all query heads attached to that KV head. Thus the optimizer follows the actual head-sharing graph rather than assuming equal Q/K tensor shapes.

With QK normalization enabled, statistics are collected after normalization and before RoPE. This is a normalization-aware local geometry, but the current implementation should not be described as the exact Jacobian pullback through QK normalization.

## 2. V/O content-flow geometry

Attention output is

```text
H = A V
Y = H W_O.
```

Q/K controls where information is read from; V/O controls what information is transported and how it is recombined.

The exact V sensitivity depends on the attention matrix A. The current implementation uses a channelwise transport proxy derived from the ratio between attended-context second moments and raw V second moments within each KV head. This estimates how strongly variation in a V channel survives attention mixing.

For W_O, the input-side metric is the second moment of the attended context H, because W_O directly acts on that representation.

V and O candidate updates are jointly Frobenius-restored after conditioning.

## 3. SwiGLU gate/up/down geometry

For

```text
z = SiLU(g) * u

g = W_gate x
u = W_up x
```

the local channel derivatives are

```text
partial z / partial u = SiLU(g)
partial z / partial g = u * SiLU'(g).
```

This gives direct diagonal sensitivity metrics:

```text
M_up,j   = E[SiLU(g_j)^2]
M_gate,j = E[(u_j * SiLU'(g_j))^2].
```

The down projection acts on z, so its input-side metric is

```text
M_down,j = E[z_j^2].
```

These three matrices form one functional family and are jointly norm-restored.

## 4. Token embedding / tied output geometry

Token rows are sampled extremely non-uniformly. ORBIT-all therefore does not treat the token embedding as an ordinary tall hidden matrix. The initial implementation uses an EMA token-frequency metric and applies a bounded inverse-frequency geometry to the AdamW candidate update, followed by norm restoration.

For tied embeddings, the same weight also serves as the LM head. The current implementation uses the language-model token stream as a shared event-frequency proxy for both roles. A richer softmax-output Fisher treatment is a planned ablation rather than an assumption baked into the first experiment.

## 5. Learned rotary frequency geometry

If rotary inverse frequencies are learned in log space,

```text
omega = exp(phi),
```

then phi is a structurally meaningful parameter even though it is not a matrix. ORBIT-all estimates frequency sensitivity from Q/K second moments and the derivative of the rotated score with respect to log frequency. The update is diagonally conditioned and norm-restored.

This specialization is intentionally limited to learned positional parameters; ORBIT-all is not trying to invent custom geometry for every scalar/vector in the model.

## 6. MoE router and expert geometry

For router probabilities

```text
p = softmax(W_r x),
```

the categorical Fisher in logit coordinates has diagonal

```text
p * (1 - p).
```

ORBIT-all uses the running mean of this quantity to condition router rows. Expert MLPs reuse the SwiGLU geometry, but their statistics are collected only on tokens actually routed to each expert, making the expert statistics conditional on routed data by construction.

The router experiment also records expert load imbalance as a diagnostic. No claim about solving load balancing is made unless the experiments support it.

## 7. Family routing

The default modes are:

```text
muon                  no function-specific geometry
orbit_qk              Q/K only
orbit_qk_vo           Q/K + V/O
orbit_qk_vo_swiglu    Q/K + V/O + SwiGLU
orbit_all              all currently implemented major families
```

Drop-one-family variants are first-class experimental modes.

## Design boundary

The project deliberately stops short of adapting every weight, bias or normalization parameter from arbitrary observed statistics. The research question is whether **large parameter families with distinct computational roles** benefit from geometry derived from those roles.
