/*
OpenCL RandomForestClassifier
classifier_class_name = ObjectSegmenter
feature_specification = gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1
num_ground_truth_dimensions = 2
num_classes = 2
num_features = 4
max_depth = 2
num_trees = 100
feature_importances = 0.7044994770448302,0.015654457326446866,0.06316698191309415,0.2166790837156289
positive_class_identifier = 1
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i0<0.6200624108314514){
 if(i2<-0.07556338608264923){
  s0+=29.0;
  s1+=3.0;
 } else {
  s0+=507.0;
 }
} else {
 if(i3<0.4496247172355652){
  s0+=3.0;
  s1+=917.0;
 } else {
  s0+=8.0;
 }
}
if(i0<0.6206761002540588){
 if(i0<0.6023684740066528){
  s0+=574.0;
 } else {
  s0+=6.0;
  s1+=2.0;
 }
} else {
 if(i3<0.4296857714653015){
  s0+=2.0;
  s1+=873.0;
 } else {
  s0+=10.0;
 }
}
if(i0<0.6206761002540588){
 if(i0<0.6023684740066528){
  s0+=564.0;
 } else {
  s0+=7.0;
  s1+=1.0;
 }
} else {
 if(i3<0.4496247172355652){
  s0+=2.0;
  s1+=887.0;
 } else {
  s0+=6.0;
 }
}
if(i0<0.6449025869369507){
 if(i0<0.6023684740066528){
  s0+=548.0;
 } else {
  s0+=8.0;
  s1+=5.0;
 }
} else {
 if(i1<0.0073423683643341064){
  s0+=7.0;
  s1+=894.0;
 } else {
  s0+=4.0;
  s1+=1.0;
 }
}
if(i0<0.6445603370666504){
 if(i0<0.5932367444038391){
  s0+=559.0;
 } else {
  s0+=6.0;
  s1+=3.0;
 }
} else {
 if(i1<-0.00791090726852417){
  s0+=4.0;
 } else {
  s0+=7.0;
  s1+=888.0;
 }
}
if(i3<0.051279351115226746){
 if(i2<0.009428739547729492){
  s0+=172.0;
  s1+=516.0;
 } else {
  s0+=13.0;
  s1+=188.0;
 }
} else {
 if(i0<0.6940549612045288){
  s0+=436.0;
  s1+=10.0;
 } else {
  s1+=132.0;
 }
}
if(i3<0.05138576775789261){
 if(i0<0.4885137677192688){
  s0+=186.0;
 } else {
  s1+=738.0;
 }
} else {
 if(i0<0.6206761002540588){
  s0+=391.0;
  s1+=1.0;
 } else {
  s0+=21.0;
  s1+=130.0;
 }
}
if(i0<0.6206761002540588){
 s0+=544.0;
} else {
 if(i2<0.09827175736427307){
  s0+=5.0;
  s1+=914.0;
 } else {
  s0+=3.0;
  s1+=1.0;
 }
}
if(i3<0.06674857437610626){
 if(i0<0.6085355877876282){
  s0+=191.0;
 } else {
  s1+=780.0;
 }
} else {
 if(i0<0.6512411832809448){
  s0+=355.0;
  s1+=1.0;
 } else {
  s0+=6.0;
  s1+=134.0;
 }
}
if(i0<0.6200624108314514){
 if(i2<-0.07556338608264923){
  s0+=26.0;
  s1+=1.0;
 } else {
  s0+=519.0;
 }
} else {
 if(i1<-0.008571237325668335){
  s0+=3.0;
 } else {
  s0+=4.0;
  s1+=914.0;
 }
}
if(i0<0.6101150512695312){
 s0+=545.0;
} else {
 if(i0<0.6940549612045288){
  s0+=18.0;
  s1+=34.0;
 } else {
  s1+=870.0;
 }
}
if(i2<-0.02777615189552307){
 if(i3<0.0726100504398346){
  s0+=2.0;
  s1+=17.0;
 } else {
  s0+=91.0;
  s1+=4.0;
 }
} else {
 if(i1<0.001041516661643982){
  s0+=410.0;
  s1+=612.0;
 } else {
  s0+=77.0;
  s1+=254.0;
 }
}
if(i0<0.6255777478218079){
 s0+=567.0;
} else {
 if(i1<-0.0038393139839172363){
  s0+=7.0;
  s1+=3.0;
 } else {
  s0+=6.0;
  s1+=884.0;
 }
}
if(i0<0.6023684740066528){
 s0+=588.0;
} else {
 if(i3<0.4296857714653015){
  s0+=5.0;
  s1+=865.0;
 } else {
  s0+=9.0;
 }
}
if(i0<0.6200624108314514){
 if(i2<-0.14327624440193176){
  s0+=20.0;
  s1+=1.0;
 } else {
  s0+=576.0;
 }
} else {
 if(i3<0.42740634083747864){
  s0+=4.0;
  s1+=860.0;
 } else {
  s0+=6.0;
 }
}
if(i0<0.6206761002540588){
 if(i2<-0.07304364442825317){
  s0+=27.0;
  s1+=3.0;
 } else {
  s0+=541.0;
 }
} else {
 if(i2<0.09827175736427307){
  s0+=6.0;
  s1+=887.0;
 } else {
  s0+=3.0;
 }
}
if(i3<0.0506649948656559){
 if(i0<0.528973400592804){
  s0+=176.0;
 } else {
  s1+=716.0;
 }
} else {
 if(i2<0.020462632179260254){
  s0+=373.0;
  s1+=54.0;
 } else {
  s0+=45.0;
  s1+=103.0;
 }
}
if(i0<0.6445603370666504){
 if(i1<-0.004651114344596863){
  s0+=25.0;
  s1+=4.0;
 } else {
  s0+=532.0;
  s1+=1.0;
 }
} else {
 if(i1<-0.003693521022796631){
  s0+=1.0;
 } else {
  s0+=4.0;
  s1+=900.0;
 }
}
if(i0<0.6200624108314514){
 if(i2<-0.07556338608264923){
  s0+=30.0;
  s1+=3.0;
 } else {
  s0+=540.0;
 }
} else {
 if(i1<0.007241636514663696){
  s0+=11.0;
  s1+=877.0;
 } else {
  s0+=6.0;
 }
}
if(i0<0.6200624108314514){
 s0+=544.0;
} else {
 if(i2<-0.046118319034576416){
  s0+=9.0;
  s1+=2.0;
 } else {
  s0+=10.0;
  s1+=902.0;
 }
}
if(i3<0.050656452775001526){
 if(i0<0.6085355877876282){
  s0+=139.0;
 } else {
  s1+=749.0;
 }
} else {
 if(i2<0.021601960062980652){
  s0+=384.0;
  s1+=55.0;
 } else {
  s0+=47.0;
  s1+=93.0;
 }
}
if(i2<-0.045612677931785583){
 if(i3<0.25185027718544006){
  s0+=11.0;
  s1+=1.0;
 } else {
  s0+=44.0;
 }
} else {
 if(i0<0.6242108345031738){
  s0+=562.0;
 } else {
  s0+=3.0;
  s1+=846.0;
 }
}
if(i2<-0.02777615189552307){
 if(i0<0.7419918179512024){
  s0+=107.0;
  s1+=4.0;
 } else {
  s1+=18.0;
 }
} else {
 if(i3<0.05652548372745514){
  s0+=166.0;
  s1+=714.0;
 } else {
  s0+=320.0;
  s1+=138.0;
 }
}
if(i0<0.6206761002540588){
 if(i0<0.600399911403656){
  s0+=569.0;
 } else {
  s0+=6.0;
  s1+=5.0;
 }
} else {
 if(i3<0.4296857714653015){
  s0+=3.0;
  s1+=879.0;
 } else {
  s0+=5.0;
 }
}
if(i3<0.05063919723033905){
 if(i2<0.01057066023349762){
  s0+=152.0;
  s1+=550.0;
 } else {
  s0+=7.0;
  s1+=154.0;
 }
} else {
 if(i0<0.6101150512695312){
  s0+=436.0;
 } else {
  s0+=18.0;
  s1+=150.0;
 }
}
if(i2<-0.0412827730178833){
 if(i3<0.24418792128562927){
  s0+=12.0;
  s1+=4.0;
 } else {
  s0+=62.0;
 }
} else {
 if(i3<0.05402521789073944){
  s0+=187.0;
  s1+=746.0;
 } else {
  s0+=316.0;
  s1+=140.0;
 }
}
if(i3<0.04614706337451935){
 if(i1<0.0007678866386413574){
  s0+=149.0;
  s1+=497.0;
 } else {
  s0+=13.0;
  s1+=200.0;
 }
} else {
 if(i1<0.0009521842002868652){
  s0+=342.0;
  s1+=67.0;
 } else {
  s0+=76.0;
  s1+=123.0;
 }
}
if(i0<0.6206761002540588){
 if(i2<-0.07406531274318695){
  s0+=40.0;
  s1+=3.0;
 } else {
  s0+=517.0;
 }
} else {
 if(i1<-0.008571237325668335){
  s0+=4.0;
 } else {
  s0+=5.0;
  s1+=898.0;
 }
}
if(i3<0.05656233802437782){
 if(i3<0.02963690459728241){
  s0+=98.0;
  s1+=540.0;
 } else {
  s0+=102.0;
  s1+=217.0;
 }
} else {
 if(i1<0.0010435432195663452){
  s0+=307.0;
  s1+=42.0;
 } else {
  s0+=65.0;
  s1+=96.0;
 }
}
if(i2<-0.04495343565940857){
 if(i1<-0.011327594518661499){
  s0+=31.0;
 } else {
  s0+=36.0;
  s1+=4.0;
 }
} else {
 if(i0<0.625124990940094){
  s0+=510.0;
 } else {
  s0+=6.0;
  s1+=880.0;
 }
}
if(i2<-0.03894422948360443){
 if(i0<0.5932367444038391){
  s0+=67.0;
 } else {
  s0+=13.0;
  s1+=8.0;
 }
} else {
 if(i3<0.04773963987827301){
  s0+=160.0;
  s1+=736.0;
 } else {
  s0+=343.0;
  s1+=140.0;
 }
}
if(i2<-0.033379361033439636){
 if(i3<0.0319942943751812){
  s1+=5.0;
 } else {
  s0+=68.0;
  s1+=5.0;
 }
} else {
 if(i2<0.004715852439403534){
  s0+=351.0;
  s1+=486.0;
 } else {
  s0+=146.0;
  s1+=406.0;
 }
}
if(i0<0.6200624108314514){
 if(i0<0.6101150512695312){
  s0+=580.0;
 } else {
  s0+=5.0;
  s1+=4.0;
 }
} else {
 if(i2<-0.04891568422317505){
  s0+=5.0;
  s1+=3.0;
 } else {
  s0+=4.0;
  s1+=866.0;
 }
}
if(i0<0.6255777478218079){
 if(i0<0.600399911403656){
  s0+=565.0;
 } else {
  s0+=4.0;
  s1+=1.0;
 }
} else {
 if(i2<-0.04914325475692749){
  s0+=6.0;
 } else {
  s0+=4.0;
  s1+=887.0;
 }
}
if(i0<0.6466456651687622){
 if(i1<-0.010357871651649475){
  s0+=14.0;
  s1+=1.0;
 } else {
  s0+=539.0;
 }
} else {
 if(i3<0.4296857714653015){
  s0+=4.0;
  s1+=904.0;
 } else {
  s0+=5.0;
 }
}
if(i0<0.6226956844329834){
 if(i2<-0.14327624440193176){
  s0+=16.0;
  s1+=1.0;
 } else {
  s0+=557.0;
 }
} else {
 if(i3<0.42740634083747864){
  s0+=5.0;
  s1+=883.0;
 } else {
  s0+=5.0;
 }
}
if(i0<0.6200624108314514){
 if(i1<-0.004692792892456055){
  s0+=40.0;
  s1+=2.0;
 } else {
  s0+=545.0;
 }
} else {
 if(i3<0.42740634083747864){
  s0+=1.0;
  s1+=872.0;
 } else {
  s0+=7.0;
 }
}
if(i0<0.610213041305542){
 s0+=542.0;
} else {
 if(i0<0.684586763381958){
  s0+=14.0;
  s1+=26.0;
 } else {
  s1+=885.0;
 }
}
if(i3<0.05810569226741791){
 if(i3<0.012855528853833675){
  s0+=13.0;
  s1+=164.0;
 } else {
  s0+=180.0;
  s1+=566.0;
 }
} else {
 if(i0<0.6200624108314514){
  s0+=381.0;
  s1+=2.0;
 } else {
  s0+=10.0;
  s1+=151.0;
 }
}
if(i3<0.04614175111055374){
 if(i1<0.0007678866386413574){
  s0+=160.0;
  s1+=520.0;
 } else {
  s0+=5.0;
  s1+=181.0;
 }
} else {
 if(i0<0.6206761002540588){
  s0+=422.0;
  s1+=1.0;
 } else {
  s0+=11.0;
  s1+=167.0;
 }
}
if(i3<0.0474872887134552){
 if(i0<0.47639983892440796){
  s0+=155.0;
 } else {
  s1+=730.0;
 }
} else {
 if(i2<0.029337078332901){
  s0+=401.0;
  s1+=81.0;
 } else {
  s0+=28.0;
  s1+=72.0;
 }
}
if(i2<-0.02777615189552307){
 if(i3<0.030492864549160004){
  s1+=18.0;
 } else {
  s0+=116.0;
  s1+=11.0;
 }
} else {
 if(i2<0.006162971258163452){
  s0+=336.0;
  s1+=462.0;
 } else {
  s0+=140.0;
  s1+=384.0;
 }
}
if(i3<0.047933273017406464){
 if(i1<0.0010131001472473145){
  s0+=157.0;
  s1+=565.0;
 } else {
  s0+=5.0;
  s1+=169.0;
 }
} else {
 if(i1<0.002439051866531372){
  s0+=367.0;
  s1+=99.0;
 } else {
  s0+=25.0;
  s1+=80.0;
 }
}
if(i3<0.0511874221265316){
 if(i0<0.6081932783126831){
  s0+=178.0;
 } else {
  s1+=762.0;
 }
} else {
 if(i2<0.0191010981798172){
  s0+=356.0;
  s1+=49.0;
 } else {
  s0+=42.0;
  s1+=80.0;
 }
}
if(i0<0.6196096539497375){
 if(i2<-0.07556338608264923){
  s0+=38.0;
  s1+=2.0;
 } else {
  s0+=559.0;
 }
} else {
 if(i3<0.4296857714653015){
  s0+=1.0;
  s1+=861.0;
 } else {
  s0+=6.0;
 }
}
if(i1<0.0008393675088882446){
 if(i0<0.644769549369812){
  s0+=521.0;
  s1+=6.0;
 } else {
  s0+=8.0;
  s1+=542.0;
 }
} else {
 if(i2<0.04494822025299072){
  s0+=84.0;
  s1+=230.0;
 } else {
  s0+=6.0;
  s1+=70.0;
 }
}
if(i0<0.6101150512695312){
 s0+=538.0;
} else {
 if(i2<-0.046118319034576416){
  s0+=9.0;
  s1+=5.0;
 } else {
  s0+=9.0;
  s1+=906.0;
 }
}
if(i3<0.05436407029628754){
 if(i1<-0.0011571794748306274){
  s0+=6.0;
  s1+=166.0;
 } else {
  s0+=159.0;
  s1+=567.0;
 }
} else {
 if(i0<0.6940549612045288){
  s0+=428.0;
  s1+=12.0;
 } else {
  s1+=129.0;
 }
}
if(i0<0.6206761002540588){
 if(i0<0.6024664640426636){
  s0+=584.0;
 } else {
  s0+=8.0;
  s1+=1.0;
 }
} else {
 if(i3<0.28172194957733154){
  s1+=860.0;
 } else {
  s0+=8.0;
  s1+=6.0;
 }
}
if(i3<0.05436407029628754){
 if(i0<0.6084024906158447){
  s0+=179.0;
 } else {
  s1+=733.0;
 }
} else {
 if(i1<0.001407518982887268){
  s0+=384.0;
  s1+=55.0;
 } else {
  s0+=32.0;
  s1+=84.0;
 }
}
if(i2<-0.03894422948360443){
 if(i0<0.7725890874862671){
  s0+=79.0;
  s1+=1.0;
 } else {
  s1+=3.0;
 }
} else {
 if(i0<0.6306800246238708){
  s0+=464.0;
 } else {
  s0+=7.0;
  s1+=913.0;
 }
}
if(i3<0.05810569226741791){
 if(i0<0.4869222044944763){
  s0+=159.0;
 } else {
  s1+=797.0;
 }
} else {
 if(i0<0.6923506259918213){
  s0+=366.0;
  s1+=11.0;
 } else {
  s1+=134.0;
 }
}
if(i2<-0.02410677820444107){
 if(i2<-0.045612677931785583){
  s0+=67.0;
  s1+=8.0;
 } else {
  s0+=63.0;
  s1+=27.0;
 }
} else {
 if(i2<0.010299257934093475){
  s0+=392.0;
  s1+=543.0;
 } else {
  s0+=77.0;
  s1+=290.0;
 }
}
if(i3<0.04773963987827301){
 if(i0<0.5308495759963989){
  s0+=166.0;
 } else {
  s1+=707.0;
 }
} else {
 if(i2<0.013430953025817871){
  s0+=377.0;
  s1+=60.0;
 } else {
  s0+=55.0;
  s1+=102.0;
 }
}
if(i3<0.049392759799957275){
 if(i2<0.009113617241382599){
  s0+=154.0;
  s1+=527.0;
 } else {
  s0+=8.0;
  s1+=216.0;
 }
} else {
 if(i0<0.6200624108314514){
  s0+=419.0;
  s1+=1.0;
 } else {
  s0+=6.0;
  s1+=136.0;
 }
}
if(i3<0.05652548372745514){
 if(i0<0.5287642478942871){
  s0+=181.0;
 } else {
  s1+=742.0;
 }
} else {
 if(i1<0.001407518982887268){
  s0+=361.0;
  s1+=55.0;
 } else {
  s0+=52.0;
  s1+=76.0;
 }
}
if(i3<0.0511874221265316){
 if(i3<0.021532859653234482){
  s0+=45.0;
  s1+=361.0;
 } else {
  s0+=129.0;
  s1+=370.0;
 }
} else {
 if(i0<0.6246635913848877){
  s0+=413.0;
  s1+=1.0;
 } else {
  s0+=12.0;
  s1+=136.0;
 }
}
if(i0<0.600399911403656){
 s0+=561.0;
} else {
 if(i3<0.2683703601360321){
  s1+=887.0;
 } else {
  s0+=12.0;
  s1+=7.0;
 }
}
if(i3<0.06556176394224167){
 if(i2<0.011069096624851227){
  s0+=201.0;
  s1+=591.0;
 } else {
  s0+=13.0;
  s1+=197.0;
 }
} else {
 if(i0<0.6923506259918213){
  s0+=361.0;
  s1+=4.0;
 } else {
  s1+=100.0;
 }
}
if(i3<0.0511874221265316){
 if(i0<0.5287642478942871){
  s0+=173.0;
 } else {
  s1+=708.0;
 }
} else {
 if(i0<0.6206761002540588){
  s0+=432.0;
 } else {
  s0+=11.0;
  s1+=143.0;
 }
}
if(i3<0.042163070291280746){
 if(i1<0.0007678866386413574){
  s0+=155.0;
  s1+=467.0;
 } else {
  s0+=5.0;
  s1+=188.0;
 }
} else {
 if(i0<0.6725754737854004){
  s0+=442.0;
  s1+=7.0;
 } else {
  s0+=4.0;
  s1+=199.0;
 }
}
if(i2<-0.02777615189552307){
 if(i0<0.5927843451499939){
  s0+=95.0;
 } else {
  s0+=8.0;
  s1+=19.0;
 }
} else {
 if(i3<0.056404076516628265){
  s0+=159.0;
  s1+=736.0;
 } else {
  s0+=313.0;
  s1+=137.0;
 }
}
if(i0<0.6252772808074951){
 if(i2<-0.07556338608264923){
  s0+=30.0;
  s1+=2.0;
 } else {
  s0+=510.0;
 }
} else {
 if(i1<-0.0035902857780456543){
  s0+=3.0;
 } else {
  s0+=7.0;
  s1+=915.0;
 }
}
if(i0<0.6252772808074951){
 if(i2<-0.07556338608264923){
  s0+=26.0;
  s1+=1.0;
 } else {
  s0+=522.0;
 }
} else {
 if(i3<0.2797892689704895){
  s1+=901.0;
 } else {
  s0+=8.0;
  s1+=9.0;
 }
}
if(i0<0.6445603370666504){
 if(i0<0.6246635913848877){
  s0+=550.0;
 } else {
  s0+=3.0;
  s1+=1.0;
 }
} else {
 if(i3<0.3040681481361389){
  s1+=895.0;
 } else {
  s0+=13.0;
  s1+=5.0;
 }
}
if(i0<0.6255777478218079){
 s0+=558.0;
} else {
 if(i3<0.2797892689704895){
  s1+=887.0;
 } else {
  s0+=13.0;
  s1+=9.0;
 }
}
if(i0<0.6206761002540588){
 s0+=546.0;
} else {
 if(i2<-0.046118319034576416){
  s0+=7.0;
  s1+=5.0;
 } else {
  s0+=6.0;
  s1+=903.0;
 }
}
if(i0<0.6200624108314514){
 if(i2<-0.07556338608264923){
  s0+=25.0;
  s1+=1.0;
 } else {
  s0+=540.0;
 }
} else {
 if(i2<-0.046118319034576416){
  s0+=3.0;
  s1+=3.0;
 } else {
  s1+=895.0;
 }
}
if(i0<0.6242108345031738){
 if(i0<0.600399911403656){
  s0+=574.0;
 } else {
  s0+=6.0;
  s1+=4.0;
 }
} else {
 if(i3<0.2797892689704895){
  s1+=865.0;
 } else {
  s0+=12.0;
  s1+=6.0;
 }
}
if(i0<0.6101150512695312){
 s0+=574.0;
} else {
 if(i3<0.311678409576416){
  s1+=872.0;
 } else {
  s0+=15.0;
  s1+=6.0;
 }
}
if(i0<0.6183962821960449){
 if(i0<0.5927843451499939){
  s0+=568.0;
 } else {
  s0+=2.0;
  s1+=1.0;
 }
} else {
 if(i3<0.2736261785030365){
  s1+=881.0;
 } else {
  s0+=12.0;
  s1+=3.0;
 }
}
if(i3<0.050656452775001526){
 if(i0<0.528973400592804){
  s0+=176.0;
 } else {
  s1+=732.0;
 }
} else {
 if(i0<0.6693422794342041){
  s0+=394.0;
  s1+=11.0;
 } else {
  s0+=3.0;
  s1+=151.0;
 }
}
if(i2<-0.03330118954181671){
 if(i0<0.7435122728347778){
  s0+=98.0;
  s1+=5.0;
 } else {
  s1+=5.0;
 }
} else {
 if(i3<0.04941509664058685){
  s0+=178.0;
  s1+=691.0;
 } else {
  s0+=345.0;
  s1+=145.0;
 }
}
if(i3<0.05652548372745514){
 if(i2<0.010329477488994598){
  s0+=175.0;
  s1+=556.0;
 } else {
  s0+=12.0;
  s1+=186.0;
 }
} else {
 if(i1<0.0009591579437255859){
  s0+=331.0;
  s1+=48.0;
 } else {
  s0+=61.0;
  s1+=98.0;
 }
}
if(i3<0.051279351115226746){
 if(i2<0.010329477488994598){
  s0+=150.0;
  s1+=540.0;
 } else {
  s0+=14.0;
  s1+=177.0;
 }
} else {
 if(i0<0.6641935110092163){
  s0+=406.0;
  s1+=6.0;
 } else {
  s0+=6.0;
  s1+=168.0;
 }
}
if(i0<0.6200624108314514){
 if(i0<0.6101150512695312){
  s0+=581.0;
 } else {
  s0+=5.0;
  s1+=4.0;
 }
} else {
 if(i1<-0.008571237325668335){
  s0+=2.0;
 } else {
  s0+=4.0;
  s1+=871.0;
 }
}
if(i3<0.0506649948656559){
 if(i0<0.5328304767608643){
  s0+=166.0;
 } else {
  s1+=725.0;
 }
} else {
 if(i0<0.6206761002540588){
  s0+=387.0;
  s1+=2.0;
 } else {
  s0+=11.0;
  s1+=176.0;
 }
}
if(i0<0.6101150512695312){
 s0+=578.0;
} else {
 if(i1<-0.00791090726852417){
  s0+=3.0;
 } else {
  s0+=8.0;
  s1+=878.0;
 }
}
if(i0<0.610213041305542){
 s0+=543.0;
} else {
 if(i3<0.4496247172355652){
  s0+=3.0;
  s1+=915.0;
 } else {
  s0+=6.0;
 }
}
if(i0<0.6206761002540588){
 if(i2<-0.1555417776107788){
  s0+=9.0;
  s1+=2.0;
 } else {
  s0+=590.0;
 }
} else {
 if(i3<0.4296857714653015){
  s1+=856.0;
 } else {
  s0+=10.0;
 }
}
if(i0<0.6101150512695312){
 s0+=598.0;
} else {
 if(i3<0.4296857714653015){
  s0+=2.0;
  s1+=854.0;
 } else {
  s0+=13.0;
 }
}
if(i3<0.0506649948656559){
 if(i0<0.528973400592804){
  s0+=177.0;
 } else {
  s1+=711.0;
 }
} else {
 if(i2<0.021570682525634766){
  s0+=390.0;
  s1+=61.0;
 } else {
  s0+=39.0;
  s1+=89.0;
 }
}
if(i2<0.008090458810329437){
 if(i0<0.600399911403656){
  s0+=470.0;
 } else {
  s0+=13.0;
  s1+=510.0;
 }
} else {
 if(i3<0.055074289441108704){
  s0+=19.0;
  s1+=252.0;
 } else {
  s0+=87.0;
  s1+=116.0;
 }
}
if(i3<0.0506649948656559){
 if(i0<0.48713141679763794){
  s0+=172.0;
 } else {
  s1+=725.0;
 }
} else {
 if(i2<0.029523491859436035){
  s0+=391.0;
  s1+=69.0;
 } else {
  s0+=37.0;
  s1+=73.0;
 }
}
if(i3<0.051279351115226746){
 if(i0<0.6081932783126831){
  s0+=170.0;
 } else {
  s1+=769.0;
 }
} else {
 if(i0<0.6252772808074951){
  s0+=375.0;
  s1+=1.0;
 } else {
  s0+=9.0;
  s1+=143.0;
 }
}
if(i0<0.6206761002540588){
 if(i0<0.6101150512695312){
  s0+=579.0;
 } else {
  s0+=6.0;
  s1+=1.0;
 }
} else {
 if(i3<0.3040681481361389){
  s1+=859.0;
 } else {
  s0+=16.0;
  s1+=6.0;
 }
}
if(i3<0.051279351115226746){
 if(i0<0.6085355877876282){
  s0+=154.0;
 } else {
  s1+=750.0;
 }
} else {
 if(i0<0.6206761002540588){
  s0+=392.0;
  s1+=3.0;
 } else {
  s0+=15.0;
  s1+=153.0;
 }
}
if(i2<-0.027987539768218994){
 if(i2<-0.043281570076942444){
  s0+=66.0;
  s1+=7.0;
 } else {
  s0+=45.0;
  s1+=20.0;
 }
} else {
 if(i0<0.6308891773223877){
  s0+=459.0;
 } else {
  s0+=3.0;
  s1+=867.0;
 }
}
if(i0<0.6206761002540588){
 if(i0<0.6101150512695312){
  s0+=564.0;
 } else {
  s0+=10.0;
  s1+=2.0;
 }
} else {
 if(i0<0.6940549612045288){
  s0+=15.0;
  s1+=33.0;
 } else {
  s1+=843.0;
 }
}
if(i2<-0.02777615189552307){
 if(i3<0.08081577718257904){
  s1+=15.0;
 } else {
  s0+=83.0;
  s1+=6.0;
 }
} else {
 if(i0<0.6327654123306274){
  s0+=454.0;
 } else {
  s0+=2.0;
  s1+=907.0;
 }
}
if(i0<0.6246635913848877){
 s0+=542.0;
} else {
 if(i3<0.4317891001701355){
  s0+=2.0;
  s1+=916.0;
 } else {
  s0+=7.0;
 }
}
if(i3<0.051279351115226746){
 if(i2<-0.0399656742811203){
  s0+=5.0;
 } else {
  s0+=156.0;
  s1+=754.0;
 }
} else {
 if(i0<0.6647427082061768){
  s0+=379.0;
  s1+=8.0;
 } else {
  s0+=5.0;
  s1+=160.0;
 }
}
if(i3<0.05810569226741791){
 if(i0<0.6092902421951294){
  s0+=200.0;
 } else {
  s1+=747.0;
 }
} else {
 if(i2<0.013410702347755432){
  s0+=312.0;
  s1+=47.0;
 } else {
  s0+=57.0;
  s1+=104.0;
 }
}
if(i3<0.04219993203878403){
 if(i0<0.4872645139694214){
  s0+=120.0;
 } else {
  s1+=684.0;
 }
} else {
 if(i0<0.6206761002540588){
  s0+=445.0;
  s1+=1.0;
 } else {
  s0+=13.0;
  s1+=204.0;
 }
}
if(i2<-0.033127255737781525){
 if(i0<0.6200624108314514){
  s0+=87.0;
 } else {
  s0+=3.0;
  s1+=13.0;
 }
} else {
 if(i3<0.049392759799957275){
  s0+=159.0;
  s1+=720.0;
 } else {
  s0+=326.0;
  s1+=159.0;
 }
}
if(i3<0.053567543625831604){
 if(i0<0.5287642478942871){
  s0+=195.0;
 } else {
  s1+=746.0;
 }
} else {
 if(i0<0.6899639368057251){
  s0+=400.0;
  s1+=12.0;
 } else {
  s1+=114.0;
 }
}
if(i0<0.6206761002540588){
 if(i2<-0.0746893584728241){
  s0+=18.0;
  s1+=2.0;
 } else {
  s0+=556.0;
 }
} else {
 if(i3<0.2736261785030365){
  s1+=875.0;
 } else {
  s0+=9.0;
  s1+=7.0;
 }
}
if(i0<0.6206761002540588){
 s0+=599.0;
} else {
 if(i3<0.2736261785030365){
  s1+=845.0;
 } else {
  s0+=14.0;
  s1+=9.0;
 }
}
if(i0<0.6200624108314514){
 if(i0<0.6101150512695312){
  s0+=556.0;
 } else {
  s0+=6.0;
  s1+=3.0;
 }
} else {
 if(i3<0.4317891001701355){
  s0+=3.0;
  s1+=894.0;
 } else {
  s0+=5.0;
 }
}
if(i2<-0.03866928815841675){
 if(i0<0.6101150512695312){
  s0+=65.0;
 } else {
  s0+=9.0;
  s1+=6.0;
 }
} else {
 if(i3<0.05138576775789261){
  s0+=170.0;
  s1+=744.0;
 } else {
  s0+=331.0;
  s1+=142.0;
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
