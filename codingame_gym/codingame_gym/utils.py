import jpype
from jpype import java
import os

def start_jvm(jar_path:str):
    """
    Starts the JVM with the game JAR. ByteBuffer shared memory is always enabled.
    
    Args:
        jar_path: Path to the game JAR file
    """
    jvm_path = jpype.getDefaultJVMPath()
    
    if not jpype.isJVMStarted():
        jvm_args = [
            jvm_path,
            "-Dgame.mode=multi",
            "-Xms2g", "-Xmx8g",  # Increase heap size (was 1g/4g, now 2g/8g)
            "-XX:+UseG1GC",       # Use G1 garbage collector for better performance
            "-XX:MaxGCPauseMillis=200",  # Limit GC pauses
            "-XX:+DisableExplicitGC",     # Prevent manual GC calls
        ]
        jpype.startJVM(*jvm_args, classpath=[jar_path])

    
def convert_to_java_list_of_lists(py_list_of_lists):
    """Optimized conversion - create ArrayList directly with pre-sized capacity."""
    ArrayList = java.util.ArrayList
    outer = ArrayList(len(py_list_of_lists))
    for sublist in py_list_of_lists:
        outer.add(ArrayList(sublist))
    return outer

def convert_to_java_list_of_lists_of_lists(py_list_of_lists_of_lists):
    """Optimized conversion - avoid intermediate function calls, pre-size ArrayLists."""
    ArrayList = java.util.ArrayList
    outer = ArrayList(len(py_list_of_lists_of_lists))
    for middle_list in py_list_of_lists_of_lists:
        middle = ArrayList(len(middle_list))
        for inner_list in middle_list:
            middle.add(ArrayList(inner_list))
        outer.add(middle)
    return outer

class RedirectJavaOutput:
    def __init__(self):
        # Access Java's System class and PrintStream class
        self.System = jpype.JPackage("java").lang.System
        self.FileOutputStream = jpype.JPackage("java").io.FileOutputStream
        self.PrintStream = jpype.JPackage("java").io.PrintStream
        # Save the original System.out and System.err
        self.original_out = self.System.out
        self.original_err = self.System.err

    def __enter__(self):
        # Redirect Java's System.out and System.err to /dev/null
        devNull = self.FileOutputStream(os.devnull)
        printStream = self.PrintStream(devNull)
        self.System.setOut(printStream)
        self.System.setErr(printStream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original System.out and System.err
        self.System.setOut(self.original_out)
        self.System.setErr(self.original_err)