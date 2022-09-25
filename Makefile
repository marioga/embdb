BUILD_DIR=build

CMAKE=cmake

all: initialize compile

initialize:
	if ! test -d $(BUILD_DIR); then \
		mkdir $(BUILD_DIR); \
	fi

compile: initialize
	cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
	$(MAKE) -C $(BUILD_DIR)

debug: initialize
	cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	$(MAKE) -C $(BUILD_DIR)

clean:
	if test -d $(BUILD_DIR); then \
		$(MAKE) clean -C $(BUILD_DIR); \
		rm -rf $(BUILD_DIR); \
	fi
