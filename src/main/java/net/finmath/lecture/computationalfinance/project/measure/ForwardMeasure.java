package net.finmath.lecture.computationalfinance.project.measure;

import java.util.Arrays;

import net.finmath.exception.CalculationException;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.interestrate.models.LIBORMarketModelFromCovarianceModel.StateSpace;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

/**
 * The \( T_k \)-forward measure for the LIBOR Market Model.
 *
 * <p>
 * Under the \( T_k \)-forward measure the numeraire is the zero-coupon bond maturing at \( T_k \):
 * \[
 *   N(T_j) = P(T_j, T_k) = \prod_{l=j}^{k-1} \frac{1}{1 + \delta_l\,L_l(T_j)}, \quad j \leq k.
 * \]
 *
 * <p>
 * The corresponding drift splits into two regions:
 * <ul>
 *   <li>For \( j &lt; k \) (rates expiring before \( T_k \)):
 *       \[
 *         \mu_j(t) = -\sum_{l=j+1}^{k-1}
 *                    \frac{\delta_l}{1+\delta_l L_l(t)}\,(\lambda_j \cdot \lambda_l)
 *       \]
 *       In particular, \( L_{k-1} \) has zero drift — it is a martingale under \( \mathbb{Q}^{T_k} \).
 *   </li>
 *   <li>For \( j \geq k \) (rates expiring after \( T_k \)):
 *       \[
 *         \mu_j(t) = +\sum_{l=k}^{j}
 *                    \frac{\delta_l}{1+\delta_l L_l(t)}\,(\lambda_j \cdot \lambda_l)
 *       \]
 *    if \( t \leq T_k \) and
 *    \[
 *       \mu_j(t) =   +\sum_{l=m(t)+1}^{j}
 *                           frac{\delta_l}{1+\delta_l L_l(t)}\,(\lambda_j \cdot \lambda_l)
 *    \]
 *    if \( t > T_k \)
 *  </li>
 * </ul>
 *
 * <p>
 * The implementation uses two separate loops:
 * a <em>terminal-like</em> downward loop for \( j &lt; k \), and
 * a <em>spot-like</em> upward loop for \( j \geq k \).
 *
 * @author Felipe, GM-1, GM-2
 */
public class ForwardMeasure implements Measure {

	/** Index k in the tenor discretization — defines the measure Q^{T_k}. */
	private final int k;

	/**
	 * Creates a \( T_k \)-forward measure.
	 *
	 * @param k The tenor index such that \( T_k \) is the maturity of the numeraire bond.
	 *          Must satisfy \( 0 \leq k \leq n \).
	 */
	public ForwardMeasure(final int k) {
		this.k = k;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * The drift is computed in two passes:
	 * <ol>
	 *   <li><b>Terminal part</b> (downward loop, \( j = k{-}1 \) down to {@code firstForwardRateIndex}):
	 *       same structure as {@link TerminalMeasure} but the sum stops at \( k{-}1 \) instead of \( n{-}1 \).
	 *       Drift is computed <em>before</em> updating the sums, so \( j \) is excluded from its own sum.
	 *   </li>
	 *   <li><b>Spot part</b> (upward loop, \( j = k \) resp. \( j = m(t) +1 \)up to \( n{-}1 \)):
	 *       same structure as {@link SpotMeasure} but the sum starts fresh from \( k \).
	 *       Sums are updated <em>before</em> computing drift, so \( j \) is included in its own sum.
	 *   </li>
	 * </ol>
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final int firstForwardRateIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final StateSpace stateSpace,
			final LIBORMarketModel model) {

		final int numberOfComponents = model.getNumberOfComponents();
		final int numberOfFactors    = process.getNumberOfFactors();
		final RandomVariable zero    = model.getRandomVariableForConstant(0.0);

		// Allocate drift array; entries below firstForwardRateIndex remain null (already fixed).
		final RandomVariable[] drift = new RandomVariable[numberOfComponents];
		for(int j = firstForwardRateIndex; j < numberOfComponents; j++) {
			drift[j] = zero;
		}

		final RandomVariable[] factorLoadingsSums = new RandomVariable[numberOfFactors];

		// -------------------------------------------------------------------------
		// Part 1 — Terminal-like downward loop for j < k
		// drift[j] = -Σ_{l=j+1}^{k-1} δ_l/(1+δ_l L_l) · (λ_j · λ_l)
		// L_{k-1} has drift = 0 (empty sum), since it is a martingale under Q^{T_k}.
		// -------------------------------------------------------------------------
		Arrays.fill(factorLoadingsSums, zero);

		final int terminalLoopStart = Math.min(k - 1, numberOfComponents - 1);
		for(int j = terminalLoopStart; j >= firstForwardRateIndex; j--) {

			final double         periodLength = model.getLiborPeriodDiscretization().getTimeStep(j);
			final RandomVariable forwardRate  = realizationAtTimeIndex[j];

			// Negative weight: -δ_j / (1 + δ_j · L_j)
			RandomVariable oneStepMeasureTransform = Scalar.of(-periodLength).discount(forwardRate, periodLength);

			if(stateSpace == StateSpace.LOGNORMAL) {
				oneStepMeasureTransform = oneStepMeasureTransform.mult(forwardRate);
			}

			final RandomVariable[] factorLoading = model.getFactorLoading(process, timeIndex, j, realizationAtTimeIndex);

			// Compute drift[j] FIRST (sums only contain l > j, up to k-1)
			drift[j] = drift[j].addSumProduct(factorLoadingsSums, factorLoading);

			// Then update sums with j (used for rates below j in next iterations)
			for(int kk = 0; kk < numberOfFactors; kk++) {
				factorLoadingsSums[kk] = factorLoadingsSums[kk].addProduct(oneStepMeasureTransform, factorLoading[kk]);
			}
		}

		// -------------------------------------------------------------------------
		// Part 2 — Spot-like upward loop for j >= k
		// drift[j] = +Σ_{l=k}^{j} δ_l/(1+δ_l L_l) · (λ_j · λ_l)
		// resp.
		// drift[j] = +Σ_{l=m(t)+1}^{j} δ_l/(1+δ_l L_l) · (λ_j · λ_l)
		// -------------------------------------------------------------------------
		Arrays.fill(factorLoadingsSums, zero);   // reset — spot part starts fresh from k

		final int spotLoopStart = Math.max(k, firstForwardRateIndex);
		for(int j = spotLoopStart; j < numberOfComponents; j++) {

			final double         periodLength = model.getLiborPeriodDiscretization().getTimeStep(j);
			final RandomVariable forwardRate  = realizationAtTimeIndex[j];

			// Positive weight: +δ_j / (1 + δ_j · L_j)
			RandomVariable oneStepMeasureTransform = Scalar.of(periodLength).discount(forwardRate, periodLength);

			if(stateSpace == StateSpace.LOGNORMAL) {
				oneStepMeasureTransform = oneStepMeasureTransform.mult(forwardRate);
			}

			final RandomVariable[] factorLoading = model.getFactorLoading(process, timeIndex, j, realizationAtTimeIndex);

			// Update sums FIRST (includes j itself), then compute drift[j]
			for(int kk = 0; kk < numberOfFactors; kk++) {
				factorLoadingsSums[kk] = factorLoadingsSums[kk].addProduct(oneStepMeasureTransform, factorLoading[kk]);
			}

			drift[j] = drift[j].addSumProduct(factorLoadingsSums, factorLoading);
		}

		return drift;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * The numeraire is \( P(T_j, T_k) \), the zero-coupon bond from \( T_j \) to \( T_k \):
	 * \[
	 *   N(T_j) = \prod_{l=j}^{k-1} \frac{1}{1 + \delta_l\,L_l(T_j)}, \quad j \leq k.
	 * \]
	 * All rates are evaluated at the same simulation time \( T_j \).
	 * For \( j = k \) the product is empty and \( N(T_k) = 1 \) (bond at maturity).
	 */
	@Override
	public RandomVariable getNumeraire(
			final MonteCarloProcess process,
			final int liborTimeIndex,
			final LIBORMarketModel model) throws CalculationException {

		// P(T_k, T_k) = 1  (bond has matured)
		if(liborTimeIndex = k) {
			return model.getRandomVariableForConstant(1.0);
		}

		// Find simulation time index for T_j; use floor if T_j falls between grid points.
		int timeIndex = process.getTimeIndex(model.getLiborPeriod(liborTimeIndex));
		if(timeIndex < 0) {
			timeIndex = -timeIndex - 1;
		}

		// N(T_j) = P(T_j, T_k) — discount from T_j through each period up to T_k
		RandomVariable numeraire = model.getRandomVariableForConstant(1.0);

		for(int l = liborTimeIndex; l < k; l++) {
			final RandomVariable forwardRate  = model.getLIBOR(process, timeIndex, l);
			final double         periodLength = model.getLiborPeriodDiscretization().getTimeStep(l);

			// numeraire /= (1 + δ_l · L_l(T_j))
			numeraire = numeraire.discount(forwardRate, periodLength);
		}

		return numeraire;
	}

	/**
	 * Returns the tenor index \( k \) that defines this measure.
	 *
	 * @return The index \( k \) such that \( T_k \) is the numeraire bond maturity.
	 */
	public int getK() {
		return k;
	}
}