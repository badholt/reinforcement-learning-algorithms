package edu.jhu.Barbara.cs335.hw5.algorithm;

import edu.jhu.Barbara.cs335.hw5.data.Action;
import edu.jhu.Barbara.cs335.hw5.data.State;
import edu.jhu.Barbara.cs335.hw5.data.Terrain;
import edu.jhu.Barbara.cs335.hw5.data.WorldMap;
import edu.jhu.Barbara.cs335.hw5.util.DefaultValueHashMap;
import edu.jhu.Barbara.cs335.hw5.util.Pair;
import java.util.Map;
import java.util.Random;

public class ValueIteratingAgent implements ReinforcementLearningAgent
{
	private static final long serialVersionUID = 1L;
	
	/** A mapping between states in that world and their expected values. */
	private Map<State, Double> expectedValues;
	/** The world in which this agent is operating. */
	private WorldMap world;	
	/** The discount factor for this agent. */
	private double discountFactor;
	/** The transition function that this agent uses. */
	private TransitionFunction transitionFunction;
	/** The reward function that this agent uses. */
	private RewardFunction rewardFunction;
	/** The convergence tolerance (epsilon). */
	private double convergenceTolerance;
	/** The number of times the agent will explore a given state-action pair before giving up on it. */
	private int minimumExplorationCount;
	/** An optimistic utility estimate of unknown or scarcely-used State-Action pairs, encouraging exploration. */
	private double uOptimistic;
	/** The record of how frequently each action has been explored from each state. */
	private Map<Pair<State, Action>, Integer> visitEvents;

	/**
	 * Creates a new value iterating agent.
	 * @param world The world in which the agent will learn.
	 */
	public ValueIteratingAgent()
	{
		this.expectedValues = new DefaultValueHashMap<>(0.0);
		this.world = null;
		this.minimumExplorationCount = 0;
		this.discountFactor = 0.5;
		this.transitionFunction = null;
		this.rewardFunction = null;
		this.convergenceTolerance = 0.000000001;
		this.uOptimistic = 0.1;
		this.visitEvents = new DefaultValueHashMap<Pair<State, Action>, Integer>(0);
	}

	/** Equation 21.5 in the textbook uses this simple exploration function: */
	private Double explorationFunction(Double u, Integer n) {
		if (n < minimumExplorationCount) {
			return uOptimistic;
		} else {
			return u;
		}
	}

	private Double utilityFunction(State state, Action currentAction) {
		/** The transition model function returns a set of all possible actions following state s, along with
		 *  their probability of occurring: */
		Double currentUtility = 0.0;
		for(Pair<State, Double> sPrime : transitionFunction.transition(state, currentAction)) {
			/** The Utility, U, of each state, multiplied by its probability of occurring:  */
			Double probabilitySPrime = sPrime.getSecond();
			Double utilitySPrime = expectedValues.get(sPrime.getFirst());
			int n;
			Pair<State, Action> sa = new Pair<>(state, currentAction);
			if (visitEvents.containsKey(sa)) {
				n = visitEvents.get(sa);
			} else {
				n = 0;
				visitEvents.put(sa, n);
			}
			currentUtility += explorationFunction(probabilitySPrime * utilitySPrime, n);
		}
		return currentUtility;
	}

	@Override
	public Policy getPolicy()
	{
		return new ValuePolicy();
	}

	/**
	 * Iterate performs a single update of the estimated utilities of each
	 * state.  Return value specifies whether a termination criterion has been
	 * met.
	 */
	@Override
	public boolean iterate()
	{
		// TODO: implement value iteration; this is basically the inside of the
		// while(!done) loop.

		/** Traverses all possible states of the map, monitoring delta, or the progression to convergence as values are
		 *  continuously updated: */
		double delta = 0.0;
		int width = world.getSize().getFirst();
		int height = world.getSize().getSecond();
		for(int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				Pair<Integer, Integer> position = new Pair<>(i,j);
				if (world.getTerrain(position) != Terrain.WALL) {
					for (Action action : Action.LEGAL_ACTIONS) {
						/** Since the maximum velocity is capped at 5, we can determine all possible velocities, in
						 *  order to survey all possible states:
						 */
						for(int k = -5; k <=5; k++){
							for(int l = -5; l <= 5; l++){
								Pair<Integer, Integer> velocity =
										new Pair<>(k, l);
								State state = new State(position, velocity);
								/**Calculate the REWARD given at each state: */
								Double reward = rewardFunction.reward(state);
								/**Calculate the MAX UTILITY of each state: */
								Double maxUtility = Double.MIN_VALUE;
								for(Action actionPrime : Action.LEGAL_ACTIONS) {
									Double currentUtility = utilityFunction(state, actionPrime);
									if (currentUtility >= maxUtility || maxUtility == Double.MIN_VALUE) {
										maxUtility = currentUtility;
									}
								}
								/** Calculate the new utility: */
								Double updatedUtility = reward + (discountFactor * maxUtility);
								/** Recalculate delta by determining the difference, or convergence progression: */
								double maxDifference = Math.abs(updatedUtility - expectedValues.get(state));
								if(maxDifference > delta){
									delta = maxDifference;
								}
								/** Update the map values: */
								expectedValues.put(state, updatedUtility);
							}
						}
					}
				}
			}
		}

		/** Check delta for convergence, termination condition: */
		return delta < (convergenceTolerance * ((1 - discountFactor) / discountFactor));
	}

	public ValueIteratingAgent duplicate()
	{
		ValueIteratingAgent ret = new ValueIteratingAgent();
		ret.setConvergenceTolerance(this.convergenceTolerance);
		ret.setDiscountFactor(this.discountFactor);
		ret.setRewardFunction(this.rewardFunction);
		ret.setTransitionFunction(this.transitionFunction);
		ret.setWorld(this.world);
		ret.expectedValues.putAll(this.expectedValues);
		return ret;
	}
	
	public double getLearningFactor()
	{
		return discountFactor;
	}

	public void setDiscountFactor(double discountFactor)
	{
		this.discountFactor = discountFactor;
	}

	public TransitionFunction getTransitionFunction()
	{
		return transitionFunction;
	}

	public void setTransitionFunction(TransitionFunction transitionFunction)
	{
		this.transitionFunction = transitionFunction;
	}

	public RewardFunction getRewardFunction()
	{
		return rewardFunction;
	}

	public void setRewardFunction(RewardFunction rewardFunction)
	{
		this.rewardFunction = rewardFunction;
	}
	
	public WorldMap getWorld()
	{
		return world;
	}

	public void setWorld(WorldMap world)
	{
		this.world = world;
	}
	
	public double getConvergenceTolerance()
	{
		return convergenceTolerance;
	}

	public void setConvergenceTolerance(double convergenceTolerance)
	{
		this.convergenceTolerance = convergenceTolerance;
	}

	/**
	 * Represents a policy that this agent would produce.
	 */
	public class ValuePolicy implements Policy
	{
		private static final long serialVersionUID = 1L;
		
		private Random random = new Random();

		/**
		 * The action an agent decides to take from a given state 
		 */
		public Action decide(State state)
		{
			// TODO: this function should return an appropriate action based on
			// an exploration policy and the current estimate of expected
			// future reward.
			Double maxUtility = Double.MIN_VALUE;
			Action maxAction = null;
			for (Action currentAction : Action.LEGAL_ACTIONS) {
				Double currentUtility = utilityFunction(state, currentAction);

				if (currentUtility >= maxUtility || maxUtility == Double.MIN_VALUE) {
					maxUtility = currentUtility;
					maxAction = currentAction;
				}
			}
			/** Update the map values: */
			Pair<State, Action> executedTransition = new Pair<>(state, maxAction);
			visitEvents.put(executedTransition, visitEvents.get(executedTransition) + 1);
			return maxAction;
		}
	}
}