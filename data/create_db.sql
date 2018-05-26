
CREATE TABLE "main"."substations" (
"rdfid" TEXT, "name" TEXT, "region_id" TEXT, PRIMARY KEY ("rdfid") 
);

CREATE TABLE "main"."analog_values" (
"rdfid" TEXT, "name" TEXT, "time" INTEGER, "value" REAL, "sub_rdfid" TEXT, PRIMARY KEY ("rdfid", "time")
);

CREATE TABLE "main"."measurements" (
"rdfid" TEXT, "name" TEXT, "time" INTEGER, "value" REAL, "sub_rdfid" TEXT, PRIMARY KEY ("rdfid", "time")
);
