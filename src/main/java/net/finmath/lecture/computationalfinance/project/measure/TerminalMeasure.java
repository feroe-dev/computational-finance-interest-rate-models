package net.finmath.lecture.computationalfinance.project.measure;

import java.util.Arrays;

import net.finmath.exception.CalculationException;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

/**
 * The terminal measure for the LIBOR Market Model.
 *
 * <p>
 * Under the terminal measure the numeraire is the zero-coupon bond maturing at the
 * last tenor date \( T_n \):
 * \[
 *   N(T_j) = P(T_j, T_n) = \prod_{l=j}^{n-1} \frac{1}{1 + \delta_l\,L_l(T_j)},
 * \]
 * where all forward rates \( L_l \) are evaluated at the <em>same</em> time \( T_j \).
 *
 * <p>
 * The corresponding risk-neutral drift for the \( j \)-th forward rate is
 * \[
 *   \mu_j(t) = -\sum_{l=j+1}^{n-1}
 *              \frac{\delta_l}{1 + \delta_l\,L_l(t)}\,(\lambda_j(t) \cdot \lambda_l(t)),
 * \]
 * i.e., only rates <em>above</em> \( j \) contribute, with a negative sign.
 *
 * @author Felipe, GM-1, GM-2
 */
public class TerminalMeasure implements Measure {

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * For the terminal measure the loop runs <em>downward</em> from \( n-1 \) to
	 * {@code firstForwardRateIndex}. The drift for component \( j \) is computed
	 * <em>before</em> adding \( j \)'s own contribution to the running sum, so only
	 * rates \( l &gt; j \) appear in \( \mu_j \). The sign of the weight is negative.
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final int firstForwardRateIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final StateSpaceTransform stateSpaceTransform,
			final LIBORMarketModel model) {

		final int numberOfComponents = model.getNumberOfComponents();
		final int numberOfFactors    = process.getNumberOfFactors();
		final RandomVariable zero    = model.getRandomVariableForConstant(0.0);

		// Allocate drift array; entries below firstForwardRateIndex remain null (already fixed).
		final RandomVariable[] drift = new RandomVariable[numberOfComponents];
		for(int j = firstForwardRateIndex; j < numberOfComponents; j++) {
			drift[j] = zero;
		}

		// Running sum over factor index k: Σ_{l > j} [-δ_l/(1+δ_l L_l)] · λ_{l,k}
		final RandomVariable[] factorLoadingsSums = new RandomVariable[numberOfFactors];
		Arrays.fill(factorLoadingsSums, zero);

		// Iterate DOWNWARD: for terminal measure we compute drift[j] FIRST (using current sums
		// which contain contributions from l > j only), THEN update sums with component j.
		for(int j = numberOfComponents - 1; j >= firstForwardRateIndex; j--) {

			final double         periodLength = model.getLiborPeriodDiscretization().getTimeStep(j);
			final RandomVariable forwardRate  = realizationAtTimeIndex[j];

			// Measure transform weight: -δ_j * g(L_j) / (1 + δ_j · L_j)  — negative sign for terminal measure
			// g(L) = 1 for normal, g(L) = L for log-normal — provided by the state-space plug-in.
			RandomVariable oneStepMeasureTransform = Scalar.of(-periodLength).discount(forwardRate, periodLength)
					.mult(stateSpaceTransform.getDriftWeightFactor(forwardRate, j));

			final RandomVariable[] factorLoading = model.getFactorLoading(process, timeIndex, j, realizationAtTimeIndex);

			// Step 1: drift[j] = factorLoadingsSums · λ_j  (sums contain l > j only, NOT j itself)
			drift[j] = drift[j].addSumProduct(factorLoadingsSums, factorLoading);

			// Step 2: update running sums to include component j (used for drift of rates below j)
			for(int k = 0; k < numberOfFactors; k++) {
				factorLoadingsSums[k] = factorLoadingsSums[k].addProduct(oneStepMeasureTransform, factorLoading[k]);
			}
		}

		return drift;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * The terminal numeraire is the zero-coupon bond \( P(T_j, T_n) \), computed as
	 * a product of discount factors from \( T_j \) to \( T_n \):
	 * \[
	 *   N(T_j) = \prod_{l=j}^{n-1} \frac{1}{1 + \delta_l\,L_l(T_j)}.
	 * \]
	 * All rates are evaluated at the <em>same</em> simulation time \( T_j \) (unlike the
	 * spot measure where each rate uses its own fixing date).
	 */
	@Override
	public RandomVariable getNumeraire(
			final MonteCarloProcess process,
			final int liborTimeIndex,
			final LIBORMarketModel model) throws CalculationException {

		// Find the simulation time index for T_j.
		// If T_j falls between grid points, use the floor index (-index - 1).
		int timeIndex = process.getTimeIndex(model.getLiborPeriod(liborTimeIndex));
		if(timeIndex < 0) {
			timeIndex = -timeIndex - 1;
		}

		// N(T_j) = P(T_j, T_n) — start at 1.0 and discount through each period up to T_n
		RandomVariable numeraire = model.getRandomVariableForConstant(1.0);

		final int lastIndex = model.getLiborPeriodDiscretization().getNumberOfTimeSteps() - 1;
		for(int l = liborTimeIndex; l <= lastIndex; l++) {
			final RandomVariable forwardRate  = model.getLIBOR(process, timeIndex, l);
			final double         periodLength = model.getLiborPeriodDiscretization().getTimeStep(l);

			// numeraire /= (1 + δ_l · L_l(T_j))
			numeraire = numeraire.discount(forwardRate, periodLength);
		}

		return numeraire;
	}
}