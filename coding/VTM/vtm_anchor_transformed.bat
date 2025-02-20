set hpcshare=\\dellnas2\hpc31\data\gaochangsheng

set project_name=%1
set cfg_name=%2

set width=%3
set height=%4
set QP=%5
set seq_name=%6

set mask_type=%7
set mask_network=%8
set alpha=%9

set hpcexe=%hpcshare%\bin\%project_name%
set hpccfg=%hpcshare%\cfg\%project_name%
set hpcorgyuv=%hpcshare%\DIICM\transformed_yuv\%mask_type%\%mask_network%\1.0_%alpha%
set hpcbits=%hpcshare%\DIICM\transformed_compressed\vtm_anchor\%mask_type%\%mask_network%\1.0_%alpha%\qp%QP%\bitstream
set hpcenclog=%hpcshare%\DIICM\transformed_compressed\vtm_anchor\%mask_type%\%mask_network%\1.0_%alpha%\qp%QP%\encoding_log
set hpcdeclog=%hpcshare%\DIICM\transformed_compressed\vtm_anchor\%mask_type%\%mask_network%\1.0_%alpha%\qp%QP%\decoding_log
set hpcdecyuv=%hpcshare%\DIICM\transformed_compressed\vtm_anchor\%mask_type%\%mask_network%\1.0_%alpha%\qp%QP%\rec_yuv

@rem echo %hpcexe%\EncoderApp -c %hpccfg%\%cfg_name%.cfg -i %hpcorgyuv%\%seq_name%.yuv -b %hpcbits%\%seq_name%.bin -o "" -q %QP% --ConformanceWindowMode=1 -wdt %width% -hgt %height%  -f 1 -fr 1 --InputChromaFormat=444 --InternalBitDepth=10 --InputBitDepth=8 --OutputBitDepth=8 >%hpcenclog%\%seq_name%.txt
@rem %hpcexe%\EncoderApp -c %hpccfg%\%cfg_name%.cfg -i %hpcorgyuv%\%seq_name%.yuv -b %hpcbits%\%seq_name%.bin -o "" -q %QP% --ConformanceWindowMode=1 -wdt %width% -hgt %height%  -f 1 -fr 1 --InputChromaFormat=444 --InternalBitDepth=10 --InputBitDepth=8 --OutputBitDepth=8 >%hpcenclog%\%seq_name%.txt

echo %hpcexe%\DecoderApp -b %hpcbits%\%seq_name%.bin -o %hpcdecyuv%\%seq_name%.yuv --OutputBitDepth=8 >%hpcdeclog%\%seq_name%.txt
%hpcexe%\DecoderApp -b %hpcbits%\%seq_name%.bin -o %hpcdecyuv%\%seq_name%.yuv --OutputBitDepth=8 >%hpcdeclog%\%seq_name%.txt
