' Launcher: creates a virtual environment (.venv) next to this file on the first run,
' installs the requirements into it and starts the ARPES app.
' Put this file next to main.py and requirements.txt, then double-click it.
Option Explicit

Dim shell, fso, folder, venvPy, basePy, code, minor, pyVer

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

folder = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = folder
venvPy = folder & "\.venv\Scripts\python.exe"

' Returns a command that starts a suitable Python, or "" if none is found.
' PyQt5 and onnxruntime work most reliably on Python 3.9 - 3.12, so those are tried first.
Function FindPython()
    Dim versions, v
    versions = Array("3.11", "3.12", "3.10", "3.9")
    For Each v In versions
        If shell.Run("cmd /c py -" & v & " -c ""import sys""", 0, True) = 0 Then
            FindPython = "py -" & v
            Exit Function
        End If
    Next
    If shell.Run("cmd /c python -c ""import sys""", 0, True) = 0 Then
        FindPython = "python"
    Else
        FindPython = ""
    End If
End Function

' Version of the Python inside .venv, for error messages
Function VenvPythonVersion()
    On Error Resume Next
    VenvPythonVersion = Trim(Replace(Replace(shell.Exec("""" & venvPy & """ -c ""import sys;print(sys.version.split()[0])""").StdOut.ReadAll(), vbCr, ""), vbLf, ""))
    If Err.Number <> 0 Or VenvPythonVersion = "" Then VenvPythonVersion = "unknown"
End Function

' 1) Create .venv if it does not exist yet
If Not fso.FileExists(venvPy) Then
    basePy = FindPython()
    If basePy = "" Then
        MsgBox "Python was not found. Install Python 3.11 from python.org " & _
               "(tick 'Add python.exe to PATH') and run this file again.", vbExclamation, "ARPES"
        WScript.Quit 1
    End If
    If basePy = "python" Then
        minor = shell.Exec("cmd /c python -c ""import sys;print(sys.version_info[1])""").StdOut.ReadAll()
        minor = CInt(Replace(Replace(minor, vbCr, ""), vbLf, ""))
        If minor >= 13 Then
            If MsgBox("Python 3." & minor & " was found. Some packages (PyQt5, onnxruntime) may not install on Python 3.13 and newer." & vbCrLf & _
                      "Recommended: install Python 3.11 (python.org, or in a console: winget install Python.Python.3.11)." & vbCrLf & vbCrLf & _
                      "Continue anyway?", vbYesNo + vbQuestion, "ARPES") = vbNo Then
                WScript.Quit 1
            End If
        End If
    End If
    code = shell.Run("cmd /c """ & basePy & " -m venv .venv""", 1, True)
    If code <> 0 Or Not fso.FileExists(venvPy) Then
        MsgBox "Could not create the virtual environment (exit code " & code & ").", vbExclamation, "ARPES"
        WScript.Quit 1
    End If
End If

' 2) pip install -r requirements.txt (inside .venv). The console stays open if the installation fails,
'    so the error text can be read and sent to the developer.
code = shell.Run("cmd /c """"" & venvPy & """ -m pip install -r requirements.txt" & _
                 " & if errorlevel 1 (echo. & echo INSTALLATION FAILED. Read the messages above and press any key to close & pause >nul & exit /b 1)""", 1, True)
If code <> 0 Then
    pyVer = VenvPythonVersion()
    MsgBox "Could not install the required packages (exit code " & code & ")." & vbCrLf & _
           "Python used: " & pyVer & vbCrLf & vbCrLf & _
           "If this is Python 3.13 or newer, install Python 3.11 (python.org, or: winget install Python.Python.3.11) and run this file again." & vbCrLf & _
           "Otherwise check the internet connection and send the text from the console window to the developer." & vbCrLf & vbCrLf & _
           "The broken .venv folder will be deleted so that the next run starts clean.", vbExclamation, "ARPES"
    On Error Resume Next
    fso.DeleteFolder folder & "\.venv", True
    On Error GoTo 0
    WScript.Quit 1
End If

' 2b) Optional packages (h5py: needed only to open .h5 files). A failure here is ignored.
shell.Run "cmd /c """"" & venvPy & """ -m pip install -r requirements-optional.txt""", 0, True

' 3) python main.py (console stays open, so error messages remain visible)
shell.Run "cmd /k """"" & venvPy & """ -X faulthandler main.py""", 1, False
