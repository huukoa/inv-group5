"""Flat, sequential (s,Q) textbook-style Theory tab.

Chapter 1 follows Examples 1–2 in a.pdf. Subsequent chapters are clearly
labelled as additional conceptual interpretation and sensitivity theory.
"""
from __future__ import annotations



def _heading(number: str, title: str, *, source: str = "") -> None:
    import streamlit as st
    st.markdown(f"## {number}. {title}")
    if source:
        st.caption(source)


def _sub(number: str, title: str) -> None:
    import streamlit as st
    st.markdown(f"### {number} {title}")


def render_textbook() -> None:
    import streamlit as st
    st.markdown("# Continuous-Review (s, Q) — Theory")
    st.markdown(
        "A continuous, textbook-style derivation of the lecture's **single-item, "
        "stochastic-demand (s, Q) model**. The mathematical foundation follows "
        "Examples 1 and 2 of *a.pdf* (pages 1–3); later chapters are explicitly "
        "identified as further explanation or sensitivity analysis."
    )
    st.caption("Scope: constant lead time · continuous review · backorders · the lecture's approximate cost objective.")

    _heading("1", "Mathematical Foundations", source="SOURCE: handwritten Examples 1–2, a.pdf, pp. 1–3")
    _sub("1.1", "Demand over a constant lead time")
    st.markdown(
        "Let **D** denote mean demand per accounting period, **L** the fixed "
        "replenishment lead time measured in that period, and **X** the random "
        "lead-time demand. Demand increments are assumed independent and stationary."
    )
    st.latex(r"\theta=E[X]=DL")
    st.latex(r"\sigma_X^2=L\sigma_D^2,\qquad \sigma_X=\sqrt{L\sigma_D^2}")
    st.markdown(
        "For Normal demand, the lead-time distribution is "
        r"$X\sim N(\theta,\sigma_X^2)$. "
        "If the base unit is a month and cost is annual, convert the demand rate "
        "and variance rate before multiplying by L. **Variance**, not standard "
        "deviation, scales linearly with the number of independent periods."
    )
    st.latex(r"D=m\mu_{\mathrm{base}},\qquad \sigma_D^2=m\sigma_{\mathrm{base}}^2")
    st.latex(r"h=iv")

    _sub("1.2", "The expected cost function")
    st.markdown(
        "Let **A** be the fixed ordering cost, **h** the unit holding cost "
        "per accounting period, **p** the penalty per expected unit short, "
        "**v** the unit purchasing price, **Q** the order quantity, and **s** "
        "the reorder point. The lecture's approximation is:"
    )
    st.latex(r"Y(Q,s)=\underbrace{\frac{AD}{Q}}_{\mathrm{ordering}}+"
             r"\underbrace{h\left(\frac Q2+s-\theta\right)}_{\mathrm{holding}}+"
             r"\underbrace{\frac{pD}{Q}n(s)}_{\mathrm{shortage}}")
    st.latex(r"Y_{\mathrm{total}}(Q,s)=Y(Q,s)+Dv")
    st.markdown("The fixed purchasing term $Dv$ does not change the stationary conditions with respect to $Q$ and $s$.")

    _sub("1.3", "Expected shortage and its derivative")
    st.markdown("Define the shortage quantity at reorder point $s$ as the positive part of $X-s$:")
    st.latex(r"(X-s)^+=\max(X-s,0)")
    st.latex(r"n(s)=E[(X-s)^+]=\int_s^\infty (x-s)g(x)\,dx")
    st.markdown("Differentiate **with respect to $s$**, applying Leibniz's rule. The lower-bound term vanishes because $x-s=0$ at $x=s$:")
    st.latex(r"\frac{dn(s)}{ds}=-\int_s^\infty g(x)\,dx")
    st.latex(r"\boxed{\frac{dn(s)}{ds}=-P(X>s)=G(s)-1}")
    st.markdown("The derivative depends on the survival probability and is valid for the continuous demand distributions considered here.")

    _sub("1.4", "First-order condition with respect to s")
    st.markdown("Differentiate $Y(Q,s)$ **partially with respect to $s$**, treating $Q$ and all cost parameters as constants:")
    st.latex(r"\frac{\partial Y(Q,s)}{\partial s}="
             r"h+\frac{pD}{Q}\frac{dn(s)}{ds}")
    st.latex(r"=h-\frac{pD}{Q}[1-G(s)]")
    st.markdown("Set this partial derivative to zero at an interior stationary point:")
    st.latex(r"\boxed{1-G(s)=\frac{Qh}{pD}}")

    _sub("1.5", "Normal standardization and expected shortage")
    st.markdown("For Normal lead-time demand, standardize to the unit Normal distribution:")
    st.latex(r"k=\frac{s-\theta}{\sigma_X},\qquad G(s)=\Phi(k),\qquad s=\theta+\sigma_X k")
    st.latex(r"1-\Phi(k)=\frac{Qh}{pD}")
    st.markdown(r"To evaluate the shortage integral, substitute $x=\theta+\sigma_X z$, $dx=\sigma_X dz$, and $s=\theta+\sigma_X k$:")
    st.latex(r"n(s)=\sigma_X\int_k^\infty (z-k)\phi(z)\,dz")
    st.latex(r"=\sigma_X\left[\int_k^\infty z\phi(z)\,dz-k\int_k^\infty\phi(z)\,dz\right]")
    st.markdown(r"For the unit Normal density, $\phi'(z)=-z\phi(z)$; therefore:")
    st.latex(r"\int_k^\infty z\phi(z)\,dz=\phi(k),\qquad\int_k^\infty\phi(z)\,dz=1-\Phi(k)")
    st.latex(r"\boxed{n(s)=\sigma_X\left[\phi(k)-k(1-\Phi(k))\right]}")

    _sub("1.6", "First-order condition with respect to Q")
    st.markdown("Differentiate $Y(Q,s)$ **partially with respect to $Q$**, holding $s$ fixed, so $n(s)$ is constant:")
    st.latex(r"\frac{\partial Y(Q,s)}{\partial Q}="
             r"-\frac{AD}{Q^2}+\frac h2-\frac{pD\,n(s)}{Q^2}")
    st.latex(r"=\frac h2-\frac{D[A+pn(s)]}{Q^2}")
    st.markdown("Set the partial derivative to zero and take the positive order quantity:")
    st.latex(r"\boxed{Q=\sqrt{\frac{2D[A+pn(s)]}{h}}}")

    _sub("1.7", "EOQ initialization and iterative updates")
    st.markdown("Start from EOQ, which omits the shortage penalty, and alternate the two first-order conditions:")
    st.latex(r"\boxed{Q_0=\sqrt{\frac{2DA}{h}}}")
    st.latex(r"\alpha_i=\frac{Q_i h}{pD},\qquad k_i=\Phi^{-1}(1-\alpha_i)")
    st.latex(r"s_i=\theta+\sigma_X k_i")
    st.latex(r"n(s_i)=\sigma_X[\phi(k_i)-k_i(1-\Phi(k_i))]")
    st.latex(r"\boxed{Q_{i+1}=\sqrt{\frac{2D[A+pn(s_i)]}{h}}}")
    st.markdown("Continue until the updates stabilize. Pair the final $Q^*$ with its matching $s^*$ before interpreting the result.")
    st.caption("Worked substitutions for the two handwritten Normal examples remain in the Results & formulas and Iterations tabs.")

    _heading("2", "Other Continuous Lead-Time Demand Distributions", source="EXTENSION: same first-order framework; distribution-dependent survival and loss")
    st.markdown("The two partial-derivative equations do not change when the lead-time demand distribution changes. What changes are $G(s)$, its inverse, and $n(s)$:")
    st.latex(    r"1-G(s_i)=\frac{Q_i h}{pD}")
    st.latex(   r"Q_{i+1}=\sqrt{\frac{2D[A+pn(s_i)]}{h}}")
    _sub("2.1", "Uniform X ~ U(a,b)")
    st.latex(r"G(s)=\frac{s-a}{b-a},\qquad 1-G(s)=\frac{b-s}{b-a}\quad(a<s<b)")
    st.latex(r"s_i=b-(b-a)\frac{Q_i h}{pD}")
    st.markdown("The shortage is the triangular area under the survival function:")
    st.latex(r"n(s)=\int_s^b\frac{b-x}{b-a}\,dx=\frac{(b-s)^2}{2(b-a)}\quad(a\le s\le b)")
    _sub("2.2", "Exponential with mean β")
    st.latex(r"1-G(s)=e^{-s/\beta},\qquad s_i=-\beta\ln\left(\frac{Q_i h}{pD}\right)")
    st.latex(r"n(s)=\int_s^\infty e^{-x/\beta}\,dx=\beta e^{-s/\beta}\quad(s\ge0)")
    _sub("2.3", "Other supported continuous distributions")
    st.markdown("Gamma, Lognormal, Weibull and Triangular are also available in the app. Each has its own probability model and expected-shortage expression. The solver still uses the same continuous first-order update, provided it admits a valid interior solution.")
    st.caption("Poisson and Binomial are discrete: the project uses integer optimization instead of treating their mass functions as smooth densities.")

    _heading("3", "Operational Meaning of the Policy", source="ADDITIONAL EXPLANATION: the business mechanism, beyond handwritten arithmetic")
    st.markdown("An (s, Q) policy asks two questions: **when to reorder** (at $s$) and **how much to reorder** (fixed $Q$). The decision is based on inventory position, not only the stock visible on shelves.")
    st.latex(r"\mathrm{Inventory\ Position}=\mathrm{On\ Hand}+\mathrm{On\ Order}-\mathrm{Backorders}")
    st.markdown("With unit-sized demand transactions, an order of size $Q$ is triggered when inventory position reaches the threshold $s$. Demand continues during the constant lead time $L$.")
    st.latex(r"\mathrm{Orders\ per\ period}\approx\frac{D}{Q^*},\qquad"
             r"\mathrm{Average\ time\ between\ orders}\approx\frac{Q^*}{D}")
    st.latex(r"\mathrm{Safety\ stock}=s^*-\theta,\qquad\mathrm{Cycle\ stock}\approx\frac{Q^*}{2}")
    st.markdown("Cycle stock supports routine sales between deliveries, while safety stock protects against demand variability during replenishment.")

    _heading("4", "Cost Trade-offs and Service", source="ADDITIONAL INTERPRETATION of the same lecture cost model")
    st.markdown("Increasing $Q^*$ generally spreads the fixed ordering cost over fewer orders but raises cycle stock. Increasing $s^*$ protects against shortage events but raises the holding-stock approximation. The objective balances these effects rather than maximizing service at any cost.")
    st.latex(r"\mathrm{Stockout\ probability}=P(X>s^*)=1-G(s^*)")
    st.latex(r"\mathrm{Cycle\ service\ level}=G(s^*)")
    st.latex(r"\mathrm{Approximate\ fill\ rate}=1-\frac{n(s^*)}{Q^*}")
    st.markdown("Cycle service level counts shortage-free cycles; fill rate measures the proportion of units met immediately. Stockout does not automatically mean lost sales in this backorder model.")

    _heading("5", "Sensitivity Analysis — Why Do Q* and s* Change?", source="ADDITIONAL ANALYSIS: not derived in handwritten Examples 1–2")
    st.markdown("Sensitivity means changing **one input while keeping the other independent inputs unchanged**, then solving the same (s, Q) problem again. The fifth tab lets the reader perform this experiment. Here we explain the underlying economics and, where justified, the mathematical direction.")
    _sub("5.1", "First-order conditions at the optimum")
    st.latex(r"\frac{h}{2}-\frac{D[A+pn(s^*)]}{(Q^*)^2}=0")
    st.latex(r"h-\frac{pD}{Q^*}[1-G(s^*)]=0")
    st.markdown("These equations jointly determine $Q^*$ and $s^*$. Changing one cost parameter causes **both** optimal decisions to adjust.")
    st.markdown("The sign statements below concern **small changes around an interior strict local minimum**, with a positive density $g(s^*)$ and:")
    st.latex(r"pDg(s^*)>h")

    _sub("5.2", "Increasing ordering cost A")
    st.markdown("A larger fixed cost makes frequent small orders more expensive. Differentiating the two optimality equations with respect to **A** gives:")
    st.latex(r"\frac{\partial Q^*}{\partial A}=\frac{DpDg(s^*)}{hQ^*[pDg(s^*)-h]}>0")
    st.latex(r"\frac{\partial s^*}{\partial A}=-\frac{D}{Q^*[pDg(s^*)-h]}<0")
    st.markdown("Subtracting the two differentiated equations isolates the reorder-point sensitivity:")
    st.latex(r"\frac{h}{Q^*}\frac{\partial Q^*}{\partial A}+\frac{h}{Q^*}\frac{\partial s^*}{\partial A}=\frac{D}{(Q^*)^2}")
    st.latex(r"\frac{h}{Q^*}\frac{\partial Q^*}{\partial A}+\frac{pDg(s^*)}{Q^*}\frac{\partial s^*}{\partial A}=0")
    st.latex(r"\frac{pDg(s^*)-h}{Q^*}\frac{\partial s^*}{\partial A}=-\frac{D}{(Q^*)^2}")
    st.markdown("**Interpretation:** Order more units per replenishment; the simultaneous cost-optimal reorder point decreases locally.")

    _sub("5.3", "Increasing shortage penalty p")
    st.markdown("A higher shortage penalty increases the cost of unmet demand. Differentiating the first-order equations with respect to **p** yields:")
    st.latex(r"\frac{\partial s^*}{\partial p}=\frac{D}{pDg(s^*)-h}"
             r"\left([1-G(s^*)]-\frac{n(s^*)}{Q^*}\right)>0")
    st.markdown("The differentiated equations are:")
    st.latex(r"\frac{h}{Q^*}\frac{\partial Q^*}{\partial p}+\frac{h}{Q^*}\frac{\partial s^*}{\partial p}=\frac{D\,n(s^*)}{(Q^*)^2}")
    st.latex(r"\frac{h}{Q^*}\frac{\partial Q^*}{\partial p}+\frac{pDg(s^*)}{Q^*}\frac{\partial s^*}{\partial p}=\frac{D[1-G(s^*)]}{Q^*}")
    st.markdown("Subtract the two equations to isolate the derivative of s*. Its positivity follows from the optimality equations and $A>0$:")
    st.latex(r"Q^*[1-G(s^*)]=\frac{2A}{p}+2n(s^*)")
    st.markdown("But the direction of **Q*** depends on the distribution's tail shape:")
    st.latex(r"\operatorname{sign}\left(\frac{\partial Q^*}{\partial p}\right)="
             r"\operatorname{sign}\left(g(s^*)n(s^*)-[1-G(s^*)]^2\right)")
    st.markdown("**Interpretation:** The reorder point rises locally, but the optimal order quantity can rise, fall, or remain constant. Do not assert that both decisions must increase.")

    _sub("5.4", "Increasing holding cost h")
    st.markdown("Higher carrying cost discourages inventory held over time. Differentiating with respect to **h**:")
    st.latex(r"\frac{\partial s^*}{\partial h}=-\frac{Q^*}{2[pDg(s^*)-h]}<0")
    st.latex(r"\frac{\partial Q^*}{\partial h}="
             r"-\frac{Q^*}{2h}+\frac{Q^*}{2[pDg(s^*)-h]}")
    st.markdown("These expressions follow by differentiating the two stationary equations:")
    st.latex(r"\frac{h}{Q^*}\frac{\partial Q^*}{\partial h}+\frac{h}{Q^*}\frac{\partial s^*}{\partial h}=-\frac12")
    st.latex(r"\frac{h}{Q^*}\frac{\partial Q^*}{\partial h}+\frac{pDg(s^*)}{Q^*}\frac{\partial s^*}{\partial h}=-1")
    st.markdown("**Interpretation:** The reorder point falls locally; Q* falls for the provided Normal examples, but that is not a universal rule across all distributions and cost ratios.")

    _sub("5.5", "Demand rate, lead time and demand variability")
    st.markdown("Changes in the demand process differ from changes in costs: they also change the probability distribution of demand during lead time. For the independent-increments Normal setup:")
    st.latex(r"\theta=DL,\qquad \sigma_X^2=L\sigma_D^2")
    st.markdown("Changing **D** shifts lead-time average demand. Changing constant **L** alters the exposure interval. Changing **σ_D** alters the spread of demand without changing its mean. Recompute $Q^*$ and $s^*$ rather than imposing universal increase/decrease rules on both.")
    st.markdown("**One-factor-at-a-time** experiments help illustrate these effects, but must specify what stays constant when a demand parameter changes.")

    _sub("5.6", "Worked sensitivity — handwritten Example 1")
    st.markdown("The following scenarios change **only one** cost coefficient by +20%, while using the same Normal lead-time demand as Example 1. Numbers come from re-solving the lecture's cost model.")
    st.markdown("| One-factor scenario | Q* | s* | Safety stock |\n|:--|--:|--:|--:|\n| Original (A=15, h=1.8, p=7) | 81.323 | 112.552 | 22.091 |\n| A increases 20% | 88.543 | 111.925 | 21.464 |\n| h increases 20% | 74.948 | 111.809 | 21.347 |\n| p increases 20% | 81.118 | 113.875 | 23.414 |")
    st.markdown("A larger penalty leads to a higher reorder point, but **Q*** falls slightly in this Normal example. The fifth tab lets you repeat these comparisons for any currently selected case.")

    _sub("5.7", "Why distribution can change the sensitivity result")
    st.markdown(r"For an Exponential lead-time demand with mean $\beta$, and an interior solution with $s^*\ge0$, the shortage identity is $n(s^*)=\beta[1-G(s^*)]$. Using the two stationary equations eliminates the penalty parameter $p$:")
    st.latex(r"(Q^*)^2=\frac{2DA}{h}+2\beta Q^*")
    st.latex(r"\boxed{Q^*=\beta+\sqrt{\beta^2+\frac{2DA}{h}}}")
    st.markdown("Thus, in this special Exponential case, **Q*** is independent of $p$, even though $s^*$ changes with $p$. Sensitivity is **model-dependent**, not a universal rule for all distributions.")

    _sub("5.8", "Common scaling of the three cost coefficients")
    st.markdown("If A, h and p are all multiplied by the same positive constant c, the policy does not change because the entire variable cost objective is multiplied by c:")
    st.latex(r"Y_{\mathrm{new}}(Q,s)=cY(Q,s),\qquad c>0")
    st.latex(r"Q^*_{\mathrm{new}}=Q^*,\qquad s^*_{\mathrm{new}}=s^*")
    st.caption("This statement refers to the variable control cost objective, with the demand model held fixed.")

    _heading("6", "Assumptions and Interpretation Limits", source="ADDITIONAL EXPLANATION of the lecture's model scope")
    st.markdown("The lecture's results are derived for a single item under continuous review with independent, stationary demand increments, constant lead time, backorders, and the specified approximate cost function. The displayed solution is continuous for smooth distributions; actual units may need integer constraints. Poisson and Binomial require an integer search in this application.")
    st.markdown("The approximation $Q/2+s-\theta$ is not an exact physical on-hand inventory identity for every inventory state. Mathematical stationarity should not be described as a universal guarantee about every real-world stock system.")
    st.markdown("---")
    st.caption("Chapter 1: handwritten Examples 1–2. Chapters 2–6: additional theory and interpretation for this specific (s, Q) model.")
