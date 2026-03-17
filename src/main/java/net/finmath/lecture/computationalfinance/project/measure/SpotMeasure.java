package net.finmath.lecture.computationalfinance.project.measure;

import java.util.Arrays;

import net.finmath.exception.CalculationException;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

/**
 * The spot (rolling) measure for the LIBOR Market Model.
 *
 * <p>
 * Under the spot measure the numeraire is the discretely compounded rolling bond
 * \[
 *   N(T_j) = \prod_{l=0}^{j-1} \bigl(1 + \delta_l\,L_l(T_l)\bigr),
 * \]
 * where each forward rate \( L_l \) is evaluated at its own fixing date \( T_l \).
 *
 * <p>
 * The corresponding risk-neutral drift for the \( j \)-th forward rate is
 * \[
 *   \mu_j(t) = \sum_{l=m(t)+1}^{j}
 *              \frac{\delta_l}{1 + \delta_l\,L_l(t)}\,(\lambda_j(t) \cdot \lambda_l(t)),
 * \]
 * where \( m(t) \) is the index of the current period and
 * \( \lambda_j \) are the factor loadings of \( L_j \).
 *
 * @author Felipe, GM-1, GM-2
 */
public class SpotMeasure implements Measure {

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * For the spot measure the drift of component \( j \) accumulates contributions
	 * from all live rates \( l \leq j \) using a running sum of factor loadings,
	 * iterating upward from {@code firstForwardRateIndex} to \( n-1 \).
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final int firstForwardRateIndex, // index of the first forward rate that has not yet fixed
			final RandomVariable[] realizationAtTimeIndex, // forward rate vector L(t_i)
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

		// Running sum over factor index k: Σ_{l ≤ j} [δ_l/(1+δ_l L_l)] · λ_{l,k}
		final RandomVariable[] factorLoadingsSums = new RandomVariable[numberOfFactors];
		Arrays.fill(factorLoadingsSums, zero); // initialize zero vector

		// Iterate upward: for spot measure we accumulate sums BEFORE using them for drift[j],
		// so drift[j] picks up contributions from all l in [firstForwardRateIndex, j].
		for(int j = firstForwardRateIndex; j < numberOfComponents; j++) {

			final double         periodLength = model.getLiborPeriodDiscretization().getTimeStep(j);
			final RandomVariable forwardRate  = realizationAtTimeIndex[j];

			// Measure transform weight: δ_j * g(L_j) / (1 + δ_j · L_j)
			// g(L) = 1 for normal, g(L) = L for log-normal — provided by the state-space plug-in.
			RandomVariable oneStepMeasureTransform = Scalar.of(periodLength).discount(forwardRate, periodLength)
					.mult(stateSpaceTransform.getDriftWeightFactor(forwardRate));

			final RandomVariable[] factorLoading = model.getFactorLoading(process, timeIndex, j, realizationAtTimeIndex);

			// Step 1: update running sums to include component j
			for(int k = 0; k < numberOfFactors; k++) {
				factorLoadingsSums[k] = factorLoadingsSums[k].addProduct(oneStepMeasureTransform, factorLoading[k]);
			}

			// Step 2: drift[j] = factorLoadingsSums · λ_j  (dot product = Σ_k sums[k] * λ_{j,k})
			drift[j] = drift[j].addSumProduct(factorLoadingsSums, factorLoading);
		}

		return drift;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * The spot numeraire is built recursively:
	 * \[
	 *   N(T_0) = 1, \qquad
	 *   N(T_j) = N(T_{j-1})\cdot\bigl(1 + \delta_{j-1}\,L_{j-1}(T_{j-1})\bigr).
	 * \]
	 * Each \( L_{j-1} \) is taken at its <em>fixing date</em> \( T_{j-1} \), which is the
	 * simulation time index closest to (but not exceeding) \( T_{j-1} \).
	 */
	@Override
	public RandomVariable getNumeraire(
			final MonteCarloProcess process,
			final int liborTimeIndex,
			final LIBORMarketModel model) throws CalculationException {

		// Base case: N(T_0) = 1
		if(liborTimeIndex == 0) {
			return model.getRandomVariableForConstant(1.0);
		}

		final double periodLength = model.getLiborPeriodDiscretization().getTimeStep(liborTimeIndex - 1);

		// Find simulation time index for fixing date T_{j-1}.
		// getTimeIndex returns a negative value when T_{j-1} falls between grid points;
		// in that case use the floor index (-index - 1).
		int fixingTimeIndex = process.getTimeIndex(model.getLiborPeriod(liborTimeIndex - 1));
		if(fixingTimeIndex < 0) {
			fixingTimeIndex = -fixingTimeIndex - 1;
		}

		// L_{j-1}(T_{j-1}): forward rate for period [T_{j-1}, T_j] evaluated at T_{j-1}
		final RandomVariable forwardRate = model.getLIBOR(process, fixingTimeIndex, liborTimeIndex - 1);

		// N(T_j) = N(T_{j-1}) * (1 + δ_{j-1} * L_{j-1}(T_{j-1}))
		return getNumeraire(process, liborTimeIndex - 1, model).accrue(forwardRate, periodLength);
	}
}
