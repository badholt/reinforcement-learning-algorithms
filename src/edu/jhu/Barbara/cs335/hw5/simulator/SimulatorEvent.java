package edu.jhu.Barbara.cs335.hw5.simulator;

/**
 * Contains information regarding a simulation event.
 * @author Zachary Palmer
 */
public class SimulatorEvent
{
	/** The simulator on which the event occurred. */
	private Simulator simulator;
	/** The step which was just registered with the simulator. */
	private SimulationStep step;
	
	public SimulatorEvent(Simulator simulator, SimulationStep step)
	{
		super();
		this.simulator = simulator;
		this.step = step;
	}

	public Simulator getSimulator()
	{
		return simulator;
	}

	public SimulationStep getStep()
	{
		return step;
	}
}
