@echo off
set project_name=DIICM
set cfg_name=encoder_intra_vtm
 
setlocal enabledelayedexpansion


set seqFile=image_names.txt
set widthFile=image_widths.txt
set heightFile=image_heights.txt

set mask_type=inferred
set mask_network=MaskRCNN_Res101_FPN_0.5
set alpha=0.8

set qpList=42

set /a iter=4999
echo iter=%iter%

for %%Q in (%qpList%) do (
    for /L %%I in (0,1,%iter%) do (
		for /f "delims=" %%S in ('powershell -Command "(Get-Content -Path '%seqFile%')[%%I]"') do (
			for /f "delims=" %%H in ('powershell -Command "(Get-Content -Path '%heightFile%')[%%I]"') do (
				for /f "delims=" %%W in ('powershell -Command "(Get-Content -Path '%widthFile%')[%%I]"') do (
					echo Processing QP=%%Q, Sequence=%%S, Height=%%H, Width=%%W, I=%%I
					call AddOne_transformed.bat %project_name% %cfg_name% %%W %%H %%Q %%S %mask_type% %mask_network% %alpha%
				)
            )
        )
    )
)
endlocal
pause