package edu.jhu.Barbara.cs335.hw5.shell.command;

import edu.jhu.Barbara.cs335.hw5.shell.Command;
import edu.jhu.Barbara.cs335.hw5.shell.Shell;

public class QuitCommand extends Command
{
	@Override
	public void execute(Shell shell, String[] args)
		throws CommandFailureException
	{
		checkArgCount(args,0,0);
		shell.setTerminate(true);
	}

	@Override
	public String getLongHelp(String name)
	{
		return "Terminates the shell.";
	}

	@Override
	public String getShortHelp()
	{
		return "quits the shell";
	}
	
}
