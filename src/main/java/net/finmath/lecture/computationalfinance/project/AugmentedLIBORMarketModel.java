package net.finmath.lecture.computationalfinance.project;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import net.finmath.exception.CalculationException;
import net.finmath.lecture.computationalfinance.project.measure.ForwardMeasure;
import net.finmath.lecture.computationalfinance.project.measure.Measure;
import net.finmath.lecture.computationalfinance.project.measure.SpotMeasure;
import net.finmath.lecture.computationalfinance.project.measure.TerminalMeasure;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.marketdata.model.AnalyticModel;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.interestrate.CalibrationProduct;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModelCalibrateable;
import net.finmath.montecarlo.model.AbstractProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;

/**
 * A LIBOR Market Model where both the probability measure and the state-space transform
 * are injected as plug-ins.
 *
 * <p>
 * This class is based on
 * {@code net.finmath.montecarlo.interestrate.models.LIBORMarketModelFromCovarianceModel}
 * from finmath-lib, with the following changes:
 * <ul>
 *   <li>The hardcoded {@code Measure} enum (SPOT / TERMINAL) is replaced by the
 *       {@link Measure} plug-in interface, allowing arbitrary measures to be injected
 *       at construction time.</li>
 *   <li>The hardcoded {@code StateSpace} enum (LOGNORMAL / NORMAL) is replaced by the
 *       {@link StateSpaceTransform} plug-in interface, allowing arbitrary state-space
 *       transforms to be injected at construction time.</li>
 *   <li>The {@code InterpolationMethod} enum is removed. This model always uses
 *       <b>linear interpolation</b> for fractional tenor periods.</li>
 *   <li>All log-linear interpolation code and its drift adjustment caches are removed.</li>
 * </ul>
 *
 * <p>
 * Three ready-to-use measure implementations are provided:
 * {@link SpotMeasure}, {@link TerminalMeasure}, and {@link ForwardMeasure}.
 *
 * @author Felipe, GM-1, GM-2
 * @see Measure
 * @see StateSpaceTransform
 * @see SpotMeasure
 * @see TerminalMeasure
 * @see ForwardMeasure
 */
public class AugmentedLIBORMarketModel extends AbstractProcessModel implements LIBORMarketModel {

	/** Tenor discretization T_0 &lt; T_1 &lt; ... &lt; T_n. */
	private final TimeDiscretization liborPeriodDiscretization;

	private final AnalyticModel      curveModel;
	private final ForwardCurve       forwardRateCurve;
	private final DiscountCurve      discountCurve;
	private final RandomVariableFactory randomVariableFactory;

	private LIBORCovarianceModel     covarianceModel;

	/** The plug-in measure — determines drift and numeraire. */
	private final Measure            measure;

	/** The plug-in state-space transform — determines initial state, transform, and Itô correction. */
	private final StateSpaceTransform stateSpaceTransform;

	private final double             liborCap;

	// Integrated covariance cache
	private double[][][]             integratedLIBORCovariance;
	private final transient Object   integratedLIBORCovarianceLazyInitLock = new Object();

	// Numeraire cache — invalidated when the process changes
	private transient MonteCarloProcess                         numerairesProcess = null;
	private transient ConcurrentHashMap<Integer, RandomVariable> numeraires       = new ConcurrentHashMap<>();
	private transient ConcurrentHashMap<Double, RandomVariable>  numeraireDiscountFactorForwardRates = new ConcurrentHashMap<>();
	private transient ConcurrentHashMap<Double, RandomVariable>  numeraireDiscountFactors            = new ConcurrentHashMap<>();

	/**
	 * Creates a LIBOR Market Model with plug-in measure and state-space transform,
	 * optionally calibrating the covariance model if calibration products are supplied.
	 *
	 * @param liborPeriodDiscretization Tenor structure \( T_0 &lt; T_1 &lt; \ldots &lt; T_n \).
	 * @param analyticModel             Analytic model for curves (may be {@code null}).
	 * @param forwardRateCurve          Initial forward rate curve.
	 * @param discountCurve             Discount curve for OIS adjustment (may be {@code null}).
	 * @param randomVariableFactory     Factory for creating {@link RandomVariable} instances.
	 * @param covarianceModel           Covariance model providing factor loadings.
	 * @param measure                   The plug-in measure (e.g. {@code new SpotMeasure()}).
	 * @param stateSpaceTransform       The plug-in state-space transform
	 *                                  (e.g. {@code new NormalStateSpaceTransform()}).
	 * @param calibrationProducts       Calibration instruments (may be empty or {@code null}).
	 * @param properties                Optional map; supports key {@code "liborCap"}.
	 * @throws CalculationException If calibration fails.
	 */
	public AugmentedLIBORMarketModel(
			final TimeDiscretization      liborPeriodDiscretization,
			final AnalyticModel           analyticModel,
			final ForwardCurve            forwardRateCurve,
			final DiscountCurve           discountCurve,
			final RandomVariableFactory   randomVariableFactory,
			final LIBORCovarianceModel    covarianceModel,
			final Measure                 measure,
			final StateSpaceTransform     stateSpaceTransform,
			final CalibrationProduct[]    calibrationProducts,
			final Map<String, ?>          properties
	) throws CalculationException {

		this.liborPeriodDiscretization = liborPeriodDiscretization;
		this.curveModel                = analyticModel;
		this.forwardRateCurve          = forwardRateCurve;
		this.discountCurve             = discountCurve;
		this.randomVariableFactory     = randomVariableFactory;
		this.measure                   = measure;
		this.stateSpaceTransform       = stateSpaceTransform;

		// Read optional properties
		double liborCapProperty = 1E5;
		if(properties != null && properties.containsKey("liborCap")) {
			liborCapProperty = (Double) properties.get("liborCap");
		}
		this.liborCap = liborCapProperty;

		// Calibrate covariance model if products are given
		if(calibrationProducts != null && calibrationProducts.length > 0) {
			LIBORCovarianceModelCalibrateable covarianceModelParametric;
			try {
				covarianceModelParametric = (LIBORCovarianceModelCalibrateable) covarianceModel;
			} catch(final Exception e) {
				throw new ClassCastException("Calibration restricted to covariance models implementing LIBORCovarianceModelCalibrateable.");
			}
			this.covarianceModel = covarianceModelParametric.getCloneCalibrated(this, calibrationProducts, null);
		} else {
			this.covarianceModel = covarianceModel;
		}
	}

	// -------------------------------------------------------------------------
	// ProcessModel interface — initial state and state-space transforms
	// -------------------------------------------------------------------------

	@Override
	public RandomVariable[] getInitialState(final MonteCarloProcess process) {
		final double[] liborInitialStates = new double[liborPeriodDiscretization.getNumberOfTimeSteps()];
		for(int i = 0; i < liborPeriodDiscretization.getNumberOfTimeSteps(); i++) {
			final double rate = forwardRateCurve.getForward(curveModel, liborPeriodDiscretization.getTime(i), liborPeriodDiscretization.getTimeStep(i));
			liborInitialStates[i] = stateSpaceTransform.getInitialState(rate);
		}
		final RandomVariable[] initialState = new RandomVariable[getNumberOfComponents()];
		for(int i = 0; i < getNumberOfComponents(); i++) {
			initialState[i] = getRandomVariableForConstant(liborInitialStates[i]);
		}
		return initialState;
	}

	@Override
	public RandomVariable applyStateSpaceTransform(final MonteCarloProcess process, final int timeIndex, final int componentIndex, final RandomVariable randomVariable) {
		RandomVariable value = stateSpaceTransform.applyTransform(randomVariable);
		if(!Double.isInfinite(liborCap)) {
			value = value.cap(liborCap);
		}
		return value;
	}

	@Override
	public RandomVariable applyStateSpaceTransformInverse(final MonteCarloProcess process, final int timeIndex, final int componentIndex, final RandomVariable randomVariable) {
		return stateSpaceTransform.applyInverseTransform(randomVariable);
	}

	@Override
	public RandomVariable getRandomVariableForConstant(final double value) {
		return randomVariableFactory.createRandomVariable(value);
	}

	// -------------------------------------------------------------------------
	// Drift — delegated to the measure plug-in
	// -------------------------------------------------------------------------

	/**
	 * Returns the drift vector \( \mu(t_i) \) by delegating to the injected {@link Measure}.
	 *
	 * <p>
	 * After the measure computes the measure-specific part, the Itô correction
	 * is added via the injected {@link StateSpaceTransform}:
	 * \( -\tfrac{1}{2}\sigma_j^2 \) for log-normal, zero for normal.
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final RandomVariable[] realizationPredictor) {

		final double time = process.getTime(timeIndex);
		int firstForwardRateIndex = this.getLiborPeriodIndex(time) + 1;
		if(firstForwardRateIndex < 0) {
			firstForwardRateIndex = -firstForwardRateIndex - 1 + 1;
		}

		// Delegate measure-specific drift to the measure plug-in
		final RandomVariable[] drift = measure.getDrift(process, timeIndex, firstForwardRateIndex, realizationAtTimeIndex, stateSpaceTransform, this);

		// Itô correction — delegated to the state-space transform plug-in
		for(int j = firstForwardRateIndex; j < getNumberOfComponents(); j++) {
			final RandomVariable variance = covarianceModel.getCovariance(time, j, j, realizationAtTimeIndex);
			drift[j] = drift[j].add(stateSpaceTransform.getItoCorrection(variance));
		}

		return drift;
	}

	@Override
	public RandomVariable[] getFactorLoading(
			final MonteCarloProcess process,
			final int timeIndex,
			final int componentIndex,
			final RandomVariable[] realizationAtTimeIndex) {
		return covarianceModel.getFactorLoading(process.getTime(timeIndex), getLiborPeriod(componentIndex), realizationAtTimeIndex);
	}

	// -------------------------------------------------------------------------
	// Numeraire — delegated to the measure plug-in
	// -------------------------------------------------------------------------

	/**
	 * Returns the numeraire \( N(t) \), delegating the at-grid calculation to
	 * the injected {@link Measure} and applying a linear interpolation for
	 * off-grid times.
	 */
	@Override
	public RandomVariable getNumeraire(final MonteCarloProcess process, final double time) throws CalculationException {

		RandomVariable numeraire = getNumeraireUnadjusted(process, time);

		// Adjust for separate discount curve (OIS discounting)
		if(discountCurve != null) {
			final RandomVariable defaultableZeroBond = getNumeraireDefaultableZeroBondAsOfTimeZero(process, time);
			final double nonDefaultableZeroBond = numeraire.invert().mult(getNumeraireUnadjusted(process, 0.0)).getAverage();
			numeraire = numeraire.mult(nonDefaultableZeroBond).div(defaultableZeroBond);
		}
		return numeraire;
	}

	/**
	 * Unadjusted numeraire (before OIS correction).
	 * For times on the tenor grid, delegates to the measure plug-in.
	 * For off-grid times, uses linear interpolation to the next tenor date.
	 */
	private RandomVariable getNumeraireUnadjusted(final MonteCarloProcess process, final double time) throws CalculationException {

		final int liborTimeIndex = getLiborPeriodIndex(time);

		if(liborTimeIndex >= 0) {
			// Time is exactly on the tenor grid — use the cached/delegated value
			return getNumeraireUnadjustedAtLIBORIndex(process, liborTimeIndex);
		}

		// Time is between two tenor dates: T_lower < time < T_upper
		final int upperIndex = -liborTimeIndex - 1;
		if(upperIndex - 1 < 0) {
			throw new IllegalArgumentException("Numeraire requested for time " + time + " before the tenor grid starts.");
		}

		// N(time) ≈ N(T_upper) / (1 + L(time, time, T_upper) * (T_upper - time))
		final RandomVariable numeraireAtUpper = getNumeraireUnadjustedAtLIBORIndex(process, upperIndex);
		final RandomVariable shortRate = getForwardRate(process, time, time, getLiborPeriod(upperIndex));
		return numeraireAtUpper.discount(shortRate, getLiborPeriod(upperIndex) - time);
	}

	/**
	 * Numeraire at a tenor grid point \( T_j \), using the measure plug-in.
	 * Results are cached per process instance.
	 */
	private RandomVariable getNumeraireUnadjustedAtLIBORIndex(final MonteCarloProcess process, final int liborTimeIndex) throws CalculationException {
		synchronized(numeraires) {
			ensureCacheConsistency(process);
			RandomVariable numeraire = numeraires.get(liborTimeIndex);
			if(numeraire == null) {
				numeraire = measure.getNumeraire(process, liborTimeIndex, this);
				numeraires.put(liborTimeIndex, numeraire);
			}
			return numeraire;
		}
	}

	// -------------------------------------------------------------------------
	// OIS discount curve adjustment (unchanged from finmath-lib)
	// -------------------------------------------------------------------------

	private RandomVariable getNumeraireDefaultableZeroBondAsOfTimeZero(final MonteCarloProcess process, final double time) {
		synchronized(numeraireDiscountFactorForwardRates) {
			ensureCacheConsistency(process);
			RandomVariable adjustment = numeraireDiscountFactors.get(time);
			if(adjustment == null) {
				// Rebuild cache from discount curve
				double dfInitial = discountCurve.getDiscountFactor(curveModel, liborPeriodDiscretization.getTime(0));
				adjustment = randomVariableFactory.createRandomVariable(dfInitial);
				numeraireDiscountFactors.put(liborPeriodDiscretization.getTime(0), adjustment);
				for(int i = 0; i < liborPeriodDiscretization.getNumberOfTimeSteps(); i++) {
					final double dfPrev    = discountCurve.getDiscountFactor(curveModel, liborPeriodDiscretization.getTime(i));
					final double dfNext    = discountCurve.getDiscountFactor(curveModel, liborPeriodDiscretization.getTime(i + 1));
					final double timeStep  = liborPeriodDiscretization.getTimeStep(i);
					final double timeNext  = liborPeriodDiscretization.getTime(i + 1);
					final RandomVariable fwd = randomVariableFactory.createRandomVariable((dfPrev / dfNext - 1.0) / timeStep);
					numeraireDiscountFactorForwardRates.put(liborPeriodDiscretization.getTime(i), fwd);
					adjustment = adjustment.discount(fwd, timeStep);
					numeraireDiscountFactors.put(timeNext, adjustment);
				}
				adjustment = numeraireDiscountFactors.get(time);
			}
			return adjustment;
		}
	}

	private void ensureCacheConsistency(final MonteCarloProcess process) {
		if(process != numerairesProcess) {
			numeraires.clear();
			numeraireDiscountFactorForwardRates.clear();
			numeraireDiscountFactors.clear();
			numerairesProcess = process;
		}
	}

	// -------------------------------------------------------------------------
	// Forward rate and LIBOR accessors
	// -------------------------------------------------------------------------

	@Override
	public RandomVariable getLIBOR(final MonteCarloProcess process, final int timeIndex, final int liborIndex) throws CalculationException {
		return process.getProcessValue(timeIndex, liborIndex);
	}

	@Override
	public RandomVariable getForwardRate(final MonteCarloProcess process, double time, final double periodStart, final double periodEnd) throws CalculationException {

		final int periodStartIndex = getLiborPeriodIndex(periodStart);
		final int periodEndIndex   = getLiborPeriodIndex(periodEnd);

		// Cap time at fixing date
		time = Math.min(time, periodStart);
		int timeIndex = process.getTimeIndex(time);
		if(timeIndex < 0) {
			timeIndex = -timeIndex - 2;
		}

		// Both start and end on grid and adjacent → model primitive
		if(periodStartIndex >= 0 && periodEndIndex >= 0 && periodStartIndex + 1 == periodEndIndex) {
			return getLIBOR(process, timeIndex, periodStartIndex);
		}

		// Interpolation on periodEnd (fractional end date — linear)
		if(periodEndIndex < 0) {
			final int    previousEndIndex = (-periodEndIndex - 1) - 1;
			final double nextEndTime      = getLiborPeriod(previousEndIndex + 1);
			final RandomVariable onePlusLongLIBORdt         = getForwardRate(process, time, periodStart, nextEndTime).mult(nextEndTime - periodStart).add(1.0);
			final RandomVariable onePlusInterpolatedLIBORDt = getOnePlusInterpolatedLIBORDt(process, timeIndex, periodEnd, previousEndIndex);
			return onePlusLongLIBORdt.div(onePlusInterpolatedLIBORDt).sub(1.0).div(periodEnd - periodStart);
		}

		// Interpolation on periodStart (fractional start date — linear)
		if(periodStartIndex < 0) {
			final int    previousStartIndex = (-periodStartIndex - 1) - 1;
			final double nextStartTime      = getLiborPeriod(previousStartIndex + 1);
			if(nextStartTime > periodEnd) {
				throw new AssertionError("Interpolation not possible.");
			}
			if(nextStartTime == periodEnd) {
				return getOnePlusInterpolatedLIBORDt(process, timeIndex, periodStart, previousStartIndex).sub(1.0).div(periodEnd - periodStart);
			}
			final RandomVariable onePlusLongLIBORdt         = getForwardRate(process, time, nextStartTime, periodEnd).mult(periodEnd - nextStartTime).add(1.0);
			final RandomVariable onePlusInterpolatedLIBORDt = getOnePlusInterpolatedLIBORDt(process, timeIndex, periodStart, previousStartIndex);
			return onePlusLongLIBORdt.mult(onePlusInterpolatedLIBORDt).sub(1.0).div(periodEnd - periodStart);
		}

		// Multi-period forward rate (compound across sub-periods)
		RandomVariable accrualAccount = null;
		for(int periodIndex = periodStartIndex; periodIndex < periodEndIndex; periodIndex++) {
			final double subPeriodLength = getLiborPeriod(periodIndex + 1) - getLiborPeriod(periodIndex);
			final RandomVariable liborOverSubPeriod = getLIBOR(process, timeIndex, periodIndex);
			accrualAccount = (accrualAccount == null)
					? liborOverSubPeriod.mult(subPeriodLength).add(1.0)
					: accrualAccount.accrue(liborOverSubPeriod, subPeriodLength);
		}
		return accrualAccount.sub(1.0).div(periodEnd - periodStart);
	}

	/**
	 * Linear interpolation of \( 1 + \delta \cdot L \) for a fractional period start/end.
	 * Only LINEAR interpolation is supported (log-linear branches have been removed).
	 */
	private RandomVariable getOnePlusInterpolatedLIBORDt(
			final MonteCarloProcess process,
			final int timeIndex,
			final double periodStartTime,
			final int liborPeriodIndex) throws CalculationException {

		final double tenorPeriodStartTime = getLiborPeriod(liborPeriodIndex);
		final double tenorPeriodEndTime   = getLiborPeriod(liborPeriodIndex + 1);
		final double tenorDt              = tenorPeriodEndTime - tenorPeriodStartTime;
		final double smallDt              = tenorPeriodEndTime - periodStartTime;
		final double alpha                = smallDt / tenorDt;

		final RandomVariable onePlusLongLIBORDt = getLIBOR(process, timeIndex, liborPeriodIndex).mult(tenorDt).add(1.0);

		// LINEAR: (1 + L*tenorDt) * alpha + (1 - alpha)
		final RandomVariable onePlusInterpolatedLIBORDt = onePlusLongLIBORDt.mult(alpha).add(1 - alpha);

		// Analytic adjustment to preserve the initial forward curve exactly
		final double analyticOnePlusLongLIBORDt         = 1 + forwardRateCurve.getForward(curveModel, tenorPeriodStartTime, tenorDt) * tenorDt;
		final double analyticOnePlusShortLIBORDt        = 1 + forwardRateCurve.getForward(curveModel, periodStartTime, smallDt) * smallDt;
		final double analyticOnePlusInterpolatedLIBORDt = analyticOnePlusLongLIBORDt * alpha + (1 - alpha);

		return onePlusInterpolatedLIBORDt.mult(analyticOnePlusShortLIBORDt / analyticOnePlusInterpolatedLIBORDt);
	}

	// -------------------------------------------------------------------------
	// Dimension and discretization accessors
	// -------------------------------------------------------------------------

	@Override
	public int getNumberOfComponents() {
		return liborPeriodDiscretization.getNumberOfTimeSteps();
	}

	@Override
	public int getNumberOfLibors() {
		return getNumberOfComponents();
	}

	@Override
	public int getNumberOfFactors() {
		return covarianceModel.getNumberOfFactors();
	}

	@Override
	public double getLiborPeriod(final int timeIndex) {
		return liborPeriodDiscretization.getTime(timeIndex);
	}

	@Override
	public int getLiborPeriodIndex(final double time) {
		return liborPeriodDiscretization.getTimeIndex(time);
	}

	@Override
	public TimeDiscretization getLiborPeriodDiscretization() {
		return liborPeriodDiscretization;
	}

	@Override
	public AnalyticModel getAnalyticModel() {
		return curveModel;
	}

	@Override
	public DiscountCurve getDiscountCurve() {
		return discountCurve;
	}

	@Override
	public ForwardCurve getForwardRateCurve() {
		return forwardRateCurve;
	}

	@Override
	public LIBORCovarianceModel getCovarianceModel() {
		return covarianceModel;
	}

	/**
	 * Returns the injected {@link Measure} plug-in.
	 *
	 * @return The measure used for drift and numeraire computation.
	 */
	public Measure getMeasure() {
		return measure;
	}

	/**
	 * Returns the injected {@link StateSpaceTransform} plug-in.
	 *
	 * @return The state-space transform used for initial state, transform and Itô correction.
	 */
	public StateSpaceTransform getStateSpaceTransform() {
		return stateSpaceTransform;
	}

	// -------------------------------------------------------------------------
	// Integrated covariance (unchanged from finmath-lib)
	// -------------------------------------------------------------------------

	@Override
	public double[][][] getIntegratedLIBORCovariance(final TimeDiscretization simulationTimeDiscretization) {
		synchronized(integratedLIBORCovarianceLazyInitLock) {
			if(integratedLIBORCovariance == null) {
				final int nTenorSteps = liborPeriodDiscretization.getNumberOfTimeSteps();
				final int nSimSteps   = simulationTimeDiscretization.getNumberOfTimeSteps();

				integratedLIBORCovariance = new double[nSimSteps][nTenorSteps][nTenorSteps];

				for(int timeIndex = 0; timeIndex < nSimSteps; timeIndex++) {
					final double dt = simulationTimeDiscretization.getTimeStep(timeIndex);
					final RandomVariable[][] factorLoadings = new RandomVariable[nTenorSteps][];
					for(int comp = 0; comp < nTenorSteps; comp++) {
						factorLoadings[comp] = covarianceModel.getFactorLoading(
								simulationTimeDiscretization.getTime(timeIndex),
								liborPeriodDiscretization.getTime(comp), null);
					}
					for(int c1 = 0; c1 < nTenorSteps; c1++) {
						for(int c2 = c1; c2 < nTenorSteps; c2++) {
							double cov = 0.0;
							if(getLiborPeriod(c1) > simulationTimeDiscretization.getTime(timeIndex)) {
								for(int f = 0; f < factorLoadings[c2].length; f++) {
									cov += factorLoadings[c1][f].doubleValue() * factorLoadings[c2][f].doubleValue() * dt;
								}
							}
							integratedLIBORCovariance[timeIndex][c1][c2] = cov;
						}
					}
				}
				// Cumulative sum over time
				for(int timeIndex = 1; timeIndex < nSimSteps; timeIndex++) {
					for(int c1 = 0; c1 < nTenorSteps; c1++) {
						for(int c2 = c1; c2 < nTenorSteps; c2++) {
							integratedLIBORCovariance[timeIndex][c1][c2] += integratedLIBORCovariance[timeIndex - 1][c1][c2];
							integratedLIBORCovariance[timeIndex][c2][c1]  = integratedLIBORCovariance[timeIndex][c1][c2];
						}
					}
				}
			}
		}
		return integratedLIBORCovariance;
	}

	// -------------------------------------------------------------------------
	// Clone support
	// -------------------------------------------------------------------------

	@Override
	public AugmentedLIBORMarketModel getCloneWithModifiedCovarianceModel(final LIBORCovarianceModel newCovarianceModel) {
		try {
			return new AugmentedLIBORMarketModel(
					liborPeriodDiscretization, curveModel, forwardRateCurve, discountCurve,
					randomVariableFactory, newCovarianceModel, measure, stateSpaceTransform, null, null);
		} catch(final CalculationException e) {
			return null;
		}
	}

	@Override
	public AugmentedLIBORMarketModel getCloneWithModifiedData(final Map<String, Object> dataModified) throws CalculationException {
		TimeDiscretization   liborPeriodDiscretization = this.liborPeriodDiscretization;
		AnalyticModel        analyticModel             = this.curveModel;
		ForwardCurve         forwardRateCurve          = this.forwardRateCurve;
		DiscountCurve        discountCurve             = this.discountCurve;
		LIBORCovarianceModel covarianceModel           = this.covarianceModel;

		if(dataModified != null) {
			liborPeriodDiscretization = (TimeDiscretization)  dataModified.getOrDefault("liborPeriodDiscretization", liborPeriodDiscretization);
			analyticModel             = (AnalyticModel)        dataModified.getOrDefault("analyticModel",             analyticModel);
			forwardRateCurve          = (ForwardCurve)         dataModified.getOrDefault("forwardRateCurve",          forwardRateCurve);
			discountCurve             = (DiscountCurve)        dataModified.getOrDefault("discountCurve",             discountCurve);
			covarianceModel           = (LIBORCovarianceModel) dataModified.getOrDefault("covarianceModel",           covarianceModel);
		}

		final Map<String, Object> properties = new HashMap<>();
		properties.put("liborCap", liborCap);

		return new AugmentedLIBORMarketModel(
				liborPeriodDiscretization, analyticModel, forwardRateCurve, discountCurve,
				randomVariableFactory, covarianceModel, measure, stateSpaceTransform, null, properties);
	}

	@Override
	public String toString() {
		return "AugmentedLIBORMarketModel"
				+ " [measure=" + measure.getClass().getSimpleName()
				+ ", stateSpaceTransform=" + stateSpaceTransform.getClass().getSimpleName()
				+ ", liborPeriodDiscretization=" + liborPeriodDiscretization + "]";
	}
}
