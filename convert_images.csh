#!/bin/csh

foreach t ('-2' '-1' '+0' '+1' '+2')
    echo ${t}
    foreach file (`ls /glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv/d01_REFL_COM_2012*-0000_*_${t}.png`)
    #foreach file (`ls /glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv/t2/d01_REFL_COM_2013*-0000_*_${t}.png`)
        set outname = `echo ${file} | cut -d'/' -f6`
        #set outname = `echo ${file} | cut -d'/' -f7`
        echo $outname

        # this convert command keeps labels along edge of image
        #convert -crop 980x1012+390+173 -colors 255 ${file} ${outname}
        #convert -crop 980x1012+390+173 -colors 255 ${file} t2_${outname}

        # this convert command strips labels from edge of image
        convert -crop 980x960+410+205 -colors 255 ${file} /glade/scratch/sobash/mode_img/${outname}
    end
end
