#!/bin/bash -e

tmpdir='_build_jar'
rm -rf $tmpdir || (exit 0)
mkdir $tmpdir
cat > $tmpdir/manifest.mf <<_EOF
Manifest-Version: 1.0
Main-Class: edu.jhu.Barbara.cs335.hw5.ReinforcementLearningMain
_EOF
mkdir $tmpdir/classes
javac -target 1.5 -sourcepath src -d $tmpdir/classes $(find src -name '*.java')
jar cfm ReinforcementLearning.jar $tmpdir/manifest.mf -C $tmpdir/classes .
rm -rf $tmpdir
