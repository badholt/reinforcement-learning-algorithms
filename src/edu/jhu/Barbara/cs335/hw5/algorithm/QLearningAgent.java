package edu.jhu.Barbara.cs335.hw5.algorithm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.jhu.Barbara.cs335.hw5.data.Action;
import edu.jhu.Barbara.cs335.hw5.data.State;
import edu.jhu.Barbara.cs335.hw5.data.WorldMap;
import edu.jhu.Barbara.cs335.hw5.simulator.SimulationStep;
import edu.jhu.Barbara.cs335.hw5.simulator.Simulator;
import edu.jhu.Barbara.cs335.hw5.simulator.SimulatorEvent;
import edu.jhu.Barbara.cs335.hw5.simulator.SimulatorListener;
import edu.jhu.Barbara.cs335.hw5.util.DefaultValueHashMap;
import edu.jhu.Barbara.cs335.hw5.util.Pair;

/**
 * A reinforcement agent which uses the Q-learning technique.
 *
 * @author Zachary Palmer
 */
public class QLearningAgent implements SimulationBasedReinforcementLearningAgent
{
	private static final long serialVersionUID = 1L;

	/** The number of times the agent will explore a given state-action pair before giving up on it. */
	private int minimumExplorationCount;
	/** The discount factor used by this agent to allow control over how important short-term gains are considered. */
	private double discountFactor;
	/** The learning factor for this agent. */
	private double learningFactor;
	/** The convergence tolerance (epsilon). */
	private double convergenceTolerance;
	/** An optimistic reward estimate of unknown or scarcely-used State-Action pairs, encouraging exploration. */
	private double rOptimistic = 0.2;

	/** Tracks the maximum change in our perception of the environment during an iteration. */
	double maximumChange = 0;

	/** The record of how frequently each action has been explored from each state. */
	private Map<Pair<State, Action>, Integer> visitEvents;
	/** The expected reward for the provided state-action pair. */
	private Map<Pair<State, Action>, Double> expectedReward;

	/** The simulator which is simulating the environment in which this agent is learning. */
	private transient Simulator simulator;

	/**
	 * General constructor.
	 */
	public QLearningAgent()
	{
		this.minimumExplorationCount = 1;
		this.discountFactor = 0.99;
		this.learningFactor = 0.5;
		this.convergenceTolerance = 0.000000001;
		this.visitEvents = new DefaultValueHashMap<Pair<State, Action>, Integer>(0);
		this.expectedReward = new DefaultValueHashMap<Pair<State, Action>, Double>(0.0);
		this.simulator = null;
	}

	@Override
	public Policy getPolicy()
	{
		return new QPolicy();
	}

	/**
	 * Iterates a single learn-as-I-go simulation for this Q learning agent. A
	 * single iteration of this algorithm will walk the agent to a goal state;
	 * thus, lower order iterations are likely to take much longer.  Return
	 * value specifies whether a termination criterion has been met.
	 */
	public boolean iterate()
	{
		// TODO: this function should call the simulator to perform a sample run
		/** Calls simulator: */
		this.simulator.simulate(this.getPolicy());

		/** Maximum convergence termination condition (Textbook, Figure 17.4) */
		Double delta = 0.0;
		return (delta < convergenceTolerance * (1 - discountFactor) / discountFactor);
	}

	@Override
	public Set<? extends SimulatorListener> getSimulatorListeners()
	{
		return Collections.singleton(new QLearningListener());
	}

	@Override
	public QLearningAgent duplicate()
	{
		QLearningAgent ret = new QLearningAgent();
		ret.setConvergenceTolerance(this.convergenceTolerance);
		ret.setDiscountFactor(this.discountFactor);
		ret.setLearningFactor(this.learningFactor);
		ret.setMinimumExplorationCount(this.minimumExplorationCount);
		ret.expectedReward.putAll(this.expectedReward);
		ret.visitEvents.putAll(this.visitEvents);
		return ret;
	}

	public int getMinimumExplorationCount()
	{
		return minimumExplorationCount;
	}

	public void setMinimumExplorationCount(int minimumExplorationCount)
	{
		this.minimumExplorationCount = minimumExplorationCount;
	}

	public double getDiscountFactor()
	{
		return discountFactor;
	}

	public void setDiscountFactor(double discountFactor)
	{
		this.discountFactor = discountFactor;
	}

	public double getLearningFactor()
	{
		return learningFactor;
	}

	public void setLearningFactor(double learningFactor)
	{
		this.learningFactor = learningFactor;
	}

	public double getConvergenceTolerance()
	{
		return convergenceTolerance;
	}

	public void setConvergenceTolerance(double convergenceTolerance)
	{
		this.convergenceTolerance = convergenceTolerance;
	}

	public Simulator getSimulator()
	{
		return simulator;
	}

	public void setSimulator(Simulator simulator)
	{
		this.simulator = simulator;
	}

	/**
	 * The policy used by this agent.
	 */
	class QPolicy implements Policy
	{
		private static final long serialVersionUID = 1L;

		/** A randomizer used to break ties. */
		private Random random = new Random();

		public QPolicy()
		{
			super();
		}

		@Override
		/**
		 * Returns the action the agent chooses to take for the given state.
		 */
		public Action decide(State state)
		{
			// TODO: this function should return an appropriate action based on
			// an exploration policy and the current estimate of expected
			// future reward.

			Action amax = null;
			Double rmax = Double.MIN_VALUE;
			int n;
			for (Action currentAction : Action.LEGAL_ACTIONS) {
				Pair<State, Action> sa = new Pair<State, Action>(state, currentAction);
				n = visitEvents.get(sa);
				Double r = 0.0;
				if (n < minimumExplorationCount) {
					r = rOptimistic;
				} else {
					r = expectedReward.get(sa);
				}

				if (r >= rmax | rmax == Double.MIN_VALUE) {
					rmax = r;
					amax = currentAction;
				}
			}
			return amax;
		}
	}

	/**
	 * The listener which learns on behalf of this agent.
	 */
	class QLearningListener implements SimulatorListener
	{
		/**
		 * Called once for every timestep of a simulation; every
		 * time an agent takes an action, an "event" occurs.
		 * Q-learning needs to do an update after every step, and this
		 * function is where it takes place.
		 */
		@Override
		public void simulationEventOccurred(SimulatorEvent event)
		{
			// TODO: this function will be called each time an action is taken;
			// this is where updates to e.g. the Q-function should occur

			/** Q(s, a): */
			State s = event.getStep().getState();
			Action a = event.getStep().getAction();
			Pair<State, Action> Qsa = new Pair<State, Action>(s, a);
			Double Q = expectedReward.get(Qsa);

			/** Q(s', a'): */
			State sPrime = event.getStep().getResultState();
			Pair<State, Action> QsaPrime = new Pair<State, Action>(sPrime, a);

			/** Qmaxa'(s', a'): */
			Double maxQPrime = Double.MIN_VALUE;
			Double rPrime = 0.0;
			for (Action aPrime : Action.LEGAL_ACTIONS) {
				/** The potential of a future state may be positive if it is a "gateway" to exploration. Just as an
				 *  optimistic value was assigned for exploration in the action selection, an optimistic value is a
				 *  potential incentive for exploration when calculating the max a' of Q(s', a'). */
				int n = visitEvents.get(aPrime);
				if (n < minimumExplorationCount) {
					maxQPrime = rOptimistic;
				} else {
					rPrime = expectedReward.get(new Pair<>(sPrime, aPrime));
				}

				/** The very first action will become the initial max, ensuring an action is always taken: */
				if (rPrime >= maxQPrime | maxQPrime == Double.MIN_VALUE) {
					maxQPrime = rPrime;
				}
			}

			/** Update the exploration value, or number of visits: */
			int n = visitEvents.get(Qsa) + 1;
			visitEvents.put(Qsa, n);

			/** The resulting reward, r: */
			Double before = event.getStep().getBeforeScore();
			Double after = event.getStep().getAfterScore();


			double update = Q + (after - before) +  learningFactor * n * (discountFactor * maxQPrime - Q);
			expectedReward.put(Qsa, update);
		}
	}
}
