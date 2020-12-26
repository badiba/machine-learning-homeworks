DebugMode = True


def Print(message):
    if (DebugMode):
        print(message)


def RaiseDataIsEmptyWarning(condition, caller):
    if (condition):
        print("Data is empty in " + caller + " function - dt.py")
