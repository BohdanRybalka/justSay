!macro JustSayStopRunningProcesses
  !insertmacro CheckIfAppIsRunning "${MAINBINARYNAME}.exe" "${PRODUCTNAME}"

  !if "${INSTALLMODE}" == "currentUser"
    nsis_tauri_utils::KillProcessCurrentUser "justsay-backend.exe"
  !else
    nsis_tauri_utils::KillProcess "justsay-backend.exe"
  !endif
  Pop $R9

  StrCpy $R8 0
  justsay_await_sidecar:
    !if "${INSTALLMODE}" == "currentUser"
      nsis_tauri_utils::FindProcessCurrentUser "justsay-backend.exe"
    !else
      nsis_tauri_utils::FindProcess "justsay-backend.exe"
    !endif
    Pop $R9
    ${If} $R9 <> 0
      Goto justsay_sidecar_gone
    ${EndIf}
    IntOp $R8 $R8 + 1
    ${If} $R8 >= 10
      Abort "JustSay's background engine (justsay-backend.exe) is still running and holds files this step must replace. Close JustSay and try again."
    ${EndIf}
    Sleep 500
    Goto justsay_await_sidecar
  justsay_sidecar_gone:
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro JustSayStopRunningProcesses
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro JustSayStopRunningProcesses
!macroend
