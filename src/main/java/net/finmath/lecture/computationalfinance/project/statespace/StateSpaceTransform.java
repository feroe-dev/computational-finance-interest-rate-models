package net.finmath.lecture.computationalfinance.project.statespace;

import net.finmath.stochastic.RandomVariable;

/**
 * Plug-in interface for the state-space transform used in a LIBOR Market Model simulation.
 *
 * <p>
 * In the Euler scheme, each forward rate \( L_j \) is simulated via an internal state
 * variable \( X_j \). The state-space transform defines the relationship between them:
 * \[
 *   L_j = f(X_j)
 * \]
 * Two standard choices are:
 * <ul>
 *   <li><b>Normal (Bachelier) state space:</b> \( f(X) = X \) (identity).
 *       The rate is simulated directly. No Itô correction is needed.</li>
 *   <li><b>Log-normal (Black) state space:</b> \( f(X) = e^X \).
 *       The log-rate is simulated. An Itô correction \( -\tfrac{1}{2}\sigma_j^2 \)
 *       must be added to the drift.</li>
 * </ul>
 *
 * <p>
 * Implementations of this interface allow the model
 * {@code AugmentedLIBORMarketModel} to support arbitrary state-space
 * transforms without any modification to the model class itself (open/closed principle).
 *
 * @author Felipe, GM-1, GM-2
 * @see NormalStateSpaceTransform
 * @see LogNormalStateSpaceTransform
 */
public interface StateSpaceTransform {

	/**
	 * Maps an initial market rate to the initial internal simulation state \( X_0 \).
	 *
	 * <p>
	 * This is the inverse of {@link #applyTransform} applied to a scalar:
	 * for log-normal state space, \( X_0 = \log(L_0) \);
	 * for normal state space, \( X_0 = L_0 \).
	 *
	 * @param rate The initial forward rate \( L_0 \) from the market curve.
	 * @return The corresponding initial internal state \( X_0 \).
	 */
	double getInitialState(double rate);

	/**
	 * Transforms an internal state \( X \) to the corresponding LIBOR rate \( L = f(X) \).
	 *
	 * <p>
	 * Called by the Euler scheme after each time step to recover the forward rate
	 * from the simulated state variable. For log-normal state space this is the
	 * exponential function; for normal state space this is the identity.
	 *
	 * @param internalState The simulated internal state \( X \).
	 * @return The LIBOR rate \( L = f(X) \).
	 */
	RandomVariable applyTransform(RandomVariable internalState);

	/**
	 * Inverse transform: maps a LIBOR rate \( L \) back to the internal state \( X = f^{-1}(L) \).
	 *
	 * <p>
	 * Used when the model needs to convert an observed rate back to the state variable,
	 * for example when re-initialising the simulation from market data.
	 * For log-normal state space this is the natural logarithm;
	 * for normal state space this is the identity.
	 *
	 * @param rate The LIBOR rate \( L \).
	 * @return The internal state \( X = f^{-1}(L) \).
	 */
	RandomVariable applyInverseTransform(RandomVariable rate);

	/**
	 * Returns the drift weight factor for the measure drift formula.
	 *
	 * <p>
	 * The drift of the \( j \)-th component under any LMM measure contains a sum of terms
	 * of the form
	 * \[
	 *   w_l \cdot (\lambda_j \cdot \lambda_l), \qquad
	 *   w_l = \frac{\pm\,\delta_l \cdot g(L_l)}{1 + \delta_l\,L_l},
	 * \]
	 * where the extra factor \( g(L_l) \) depends on the state space:
	 * <ul>
	 *   <li><b>Normal:</b> \( g(L) = 1 \) — the rate enters the numerator only once.</li>
	 *   <li><b>Log-normal:</b> \( g(L) = L \) — an additional \( L_l \) factor appears
	 *       because the SDE is written in terms of \( \log L \).</li>
	 * </ul>
	 *
	 * @param liborRate The current LIBOR rate \( L_j \).
	 * @return \( g(L_j) \): the rate itself for log-normal, or a scalar 1.0 for normal.
	 */
	RandomVariable getDriftWeightFactor(RandomVariable liborRate);

	/**
	 * Returns the Itô correction term to be added to the drift of component \( j \).
	 *
	 * <p>
	 * When the state-space transform is non-linear (e.g. log-normal), Itô's lemma
	 * introduces a correction to the drift of \( X_j \):
	 * \[
	 *   \text{correction}_j = -\tfrac{1}{2}\,\sigma_j^2
	 * \]
	 * where \( \sigma_j^2 \) is the instantaneous variance of component \( j \).
	 * For a linear (normal) transform the correction is zero.
	 *
	 * @param variance The instantaneous variance \( \sigma_j^2 = \text{Cov}(j,j) \)
	 *                 of the \( j \)-th component at the current time step.
	 * @return The Itô correction \( -\tfrac{1}{2}\sigma_j^2 \) (log-normal)
	 *         or zero (normal) as a {@link RandomVariable}.
	 */
	RandomVariable getItoCorrection(RandomVariable variance);
}